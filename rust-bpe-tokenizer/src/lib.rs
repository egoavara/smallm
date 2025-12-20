use ahash::AHashMap;
use fancy_regex::Regex;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::{BinaryHeap, HashMap};

/// Sentinel value to mark chunk boundaries (cannot be a valid token ID)
const CHUNK_BOUNDARY: u32 = u32::MAX;
/// Sentinel value for deleted/merged tokens (tombstone)
const TOMBSTONE: u32 = u32::MAX - 1;

/// GPT-4 style regex pattern for tokenization
const GPT4_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";

/// Fast BPE Tokenizer implemented in Rust
#[pyclass]
pub struct RustBPE {
    pattern: String,
    regex: Regex,
    merges: AHashMap<(u32, u32), u32>,
    vocab: AHashMap<u32, Vec<u8>>,
    special_tokens: AHashMap<String, u32>,
}

#[pymethods]
impl RustBPE {
    #[new]
    #[pyo3(signature = (pattern=None))]
    fn new(pattern: Option<String>) -> PyResult<Self> {
        let pattern = pattern.unwrap_or_else(|| GPT4_PATTERN.to_string());
        let regex = Regex::new(&pattern)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        Ok(Self {
            pattern,
            regex,
            merges: AHashMap::new(),
            vocab: AHashMap::new(),
            special_tokens: AHashMap::new(),
        })
    }

    /// Train BPE tokenizer with position indexing for O(1) pair lookup
    ///
    /// Events sent to callback (as dict):
    /// - {"event": "chunking_start"}
    /// - {"event": "chunking_done", "num_chunks": int, "num_tokens": int}
    /// - {"event": "flattening_start"}
    /// - {"event": "flattening_done", "array_size": int}
    /// - {"event": "indexing_start"}
    /// - {"event": "indexing_done", "num_pairs": int}
    /// - {"event": "heap_built", "heap_size": int}
    /// - {"event": "training_start", "num_merges": int}
    /// - {"event": "merge", "step": int, "total": int, "token": str, "count": int, "new_id": int}
    /// - {"event": "training_done", "vocab_size": int}
    #[pyo3(signature = (text, vocab_size, verbose=false, callback=None))]
    fn train(&mut self, text: &str, vocab_size: u32, verbose: bool, callback: Option<PyObject>) -> PyResult<()> {
        if vocab_size < 256 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "vocab_size must be at least 256",
            ));
        }

        let num_merges = vocab_size - 256;

        // Helper to emit events
        macro_rules! emit {
            ($cb:expr, $($key:expr => $value:expr),* $(,)?) => {
                if let Some(ref cb) = $cb {
                    Python::with_gil(|py| {
                        let dict = pyo3::types::PyDict::new_bound(py);
                        $(dict.set_item($key, $value).unwrap();)*
                        let _ = cb.call1(py, (dict,));
                    });
                }
            };
        }

        // Phase 1: Chunking (parallel by lines)
        emit!(callback, "event" => "chunking_start");
        if verbose {
            println!("Splitting text into chunks...");
        }

        let lines: Vec<&str> = text.lines().collect();
        let pattern = &self.regex;

        let chunk_results: Vec<Vec<u32>> = lines
            .par_iter()
            .flat_map(|line| {
                pattern
                    .find_iter(line)
                    .filter_map(|m| m.ok())
                    .map(|m| m.as_str().bytes().map(|b| b as u32).collect::<Vec<u32>>())
                    .collect::<Vec<_>>()
            })
            .collect();

        let num_chunks = chunk_results.len();
        let num_tokens: usize = chunk_results.iter().map(|c| c.len()).sum();
        emit!(callback, "event" => "chunking_done", "num_chunks" => num_chunks, "num_tokens" => num_tokens);

        if verbose {
            println!("Total chunks: {}, Total tokens: {}", num_chunks, num_tokens);
        }

        // Phase 2: Flatten into single array with boundary markers
        emit!(callback, "event" => "flattening_start");
        if verbose {
            println!("Flattening tokens into single array...");
        }

        let total_size = num_tokens + num_chunks.saturating_sub(1);
        let mut tokens: Vec<u32> = Vec::with_capacity(total_size);

        for (i, chunk) in chunk_results.into_iter().enumerate() {
            if i > 0 {
                tokens.push(CHUNK_BOUNDARY);
            }
            tokens.extend(chunk);
        }

        let array_size = tokens.len();
        emit!(callback, "event" => "flattening_done", "array_size" => array_size);

        if verbose {
            println!("Flat array size: {}", array_size);
        }

        // Initialize vocab
        self.vocab.clear();
        for i in 0u32..256 {
            self.vocab.insert(i, vec![i as u8]);
        }
        self.merges.clear();

        // Phase 3: Build pair → positions index (parallel)
        emit!(callback, "event" => "indexing_start");
        if verbose {
            println!("Building pair position index...");
        }

        let mut pair_positions: AHashMap<(u32, u32), Vec<usize>> = self.build_pair_index(&tokens);

        let num_pairs = pair_positions.len();
        emit!(callback, "event" => "indexing_done", "num_pairs" => num_pairs);

        if verbose {
            println!("Unique pairs: {}", num_pairs);
        }

        // Phase 4: Build heap from pair counts
        let mut heap: BinaryHeap<(i64, (u32, u32))> = pair_positions
            .iter()
            .map(|(&pair, positions)| (positions.len() as i64, pair))
            .collect();

        let heap_size = heap.len();
        emit!(callback, "event" => "heap_built", "heap_size" => heap_size);

        // Phase 5: Training loop
        emit!(callback, "event" => "training_start", "num_merges" => num_merges);
        if verbose {
            println!("Starting BPE training...");
        }

        for i in 0..num_merges {
            // Find best pair using heap (lazy deletion)
            let (pair_a, pair_b, count) = loop {
                match heap.pop() {
                    Some((heap_count, pair)) => {
                        let actual = pair_positions
                            .get(&pair)
                            .map(|p| p.len() as i64)
                            .unwrap_or(0);
                        if actual == heap_count && actual > 1 {
                            break (pair.0, pair.1, actual);
                        }
                    }
                    None => {
                        emit!(callback, "event" => "no_more_pairs", "step" => i);
                        if verbose {
                            println!("\nNo more pairs at iteration {}", i);
                        }
                        emit!(callback, "event" => "training_done", "vocab_size" => self.vocab.len());
                        return Ok(());
                    }
                }
            };

            let new_id = 256 + i;

            // Get positions to merge (take ownership)
            let positions = pair_positions.remove(&(pair_a, pair_b)).unwrap_or_default();

            // Merge at positions and update index
            self.merge_with_index(
                &mut tokens,
                &mut pair_positions,
                &positions,
                pair_a,
                pair_b,
                new_id,
                &mut heap,
            );

            // Record merge
            self.merges.insert((pair_a, pair_b), new_id);
            let mut new_bytes = self.vocab.get(&pair_a).unwrap().clone();
            new_bytes.extend(self.vocab.get(&pair_b).unwrap());
            self.vocab.insert(new_id, new_bytes.clone());

            // Emit merge event
            if let Some(ref cb) = callback {
                if i % 10 == 0 || i == num_merges - 1 {
                    Python::with_gil(|py| {
                        let dict = pyo3::types::PyDict::new_bound(py);
                        dict.set_item("event", "merge").unwrap();
                        dict.set_item("step", i).unwrap();
                        dict.set_item("total", num_merges).unwrap();
                        dict.set_item("token", String::from_utf8_lossy(&new_bytes).to_string()).unwrap();
                        dict.set_item("count", count).unwrap();
                        dict.set_item("new_id", new_id).unwrap();
                        dict.set_item("pair", (pair_a, pair_b)).unwrap();
                        let _ = cb.call1(py, (dict,));
                    });
                }
            } else if verbose && i % 100 == 0 {
                let token_str = String::from_utf8_lossy(&new_bytes);
                let display: String = token_str.chars().take(10).collect();
                println!("  [{}/{}] token={:?}, count={}", i, num_merges, display, count);
            }
        }

        emit!(callback, "event" => "training_done", "vocab_size" => self.vocab.len());
        if verbose {
            println!("\nTraining complete! Vocab size: {}", self.vocab.len());
        }

        Ok(())
    }

    /// Encode text to token IDs
    fn encode(&self, text: &str) -> PyResult<Vec<u32>> {
        let mut ids = Vec::new();

        for m in self.regex.find_iter(text).filter_map(|m| m.ok()) {
            let chunk = m.as_str();
            let chunk_ids = self.encode_chunk(chunk);
            ids.extend(chunk_ids);
        }

        Ok(ids)
    }

    /// Decode token IDs to text
    fn decode(&self, ids: Vec<u32>) -> PyResult<String> {
        let mut bytes = Vec::new();

        for id in ids {
            if let Some(token_bytes) = self.vocab.get(&id) {
                bytes.extend(token_bytes);
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Unknown token ID: {}", id),
                ));
            }
        }

        Ok(String::from_utf8_lossy(&bytes).to_string())
    }

    /// Register special tokens
    fn register_special_tokens(&mut self, tokens: HashMap<String, u32>) {
        self.special_tokens = tokens.into_iter().collect();
    }

    /// Get vocabulary size
    #[getter]
    fn vocab_size(&self) -> usize {
        self.vocab.len() + self.special_tokens.len()
    }

    /// Get merges as list of tuples (sorted by merge order)
    fn get_merges(&self) -> Vec<((u32, u32), u32)> {
        let mut merges: Vec<_> = self.merges.iter().map(|(&k, &v)| (k, v)).collect();
        merges.sort_by_key(|(_, id)| *id);
        merges
    }

    /// Set merges from list of tuples
    fn set_merges(&mut self, merges: Vec<((u32, u32), u32)>) {
        self.merges.clear();
        self.vocab.clear();

        // Initialize base vocab
        for i in 0u32..256 {
            self.vocab.insert(i, vec![i as u8]);
        }

        // Apply merges in order
        for ((a, b), new_id) in merges {
            self.merges.insert((a, b), new_id);
            let mut new_bytes = self.vocab.get(&a).unwrap_or(&vec![]).clone();
            new_bytes.extend(self.vocab.get(&b).unwrap_or(&vec![]));
            self.vocab.insert(new_id, new_bytes);
        }
    }
}

impl RustBPE {
    /// Build pair → positions index (parallel)
    /// Position stored is the index of the FIRST token of the pair
    fn build_pair_index(&self, tokens: &[u32]) -> AHashMap<(u32, u32), Vec<usize>> {
        let num_threads = rayon::current_num_threads();
        let chunk_size = (tokens.len() / (num_threads * 16)).max(10000);

        let partial_maps: Vec<AHashMap<(u32, u32), Vec<usize>>> = tokens
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                let base_offset = chunk_idx * chunk_size;
                let mut local_map: AHashMap<(u32, u32), Vec<usize>> = AHashMap::new();

                let mut i = 0;
                while i < chunk.len() {
                    let a = chunk[i];
                    if a == CHUNK_BOUNDARY || a == TOMBSTONE {
                        i += 1;
                        continue;
                    }

                    // Find next non-tombstone token
                    let mut j = i + 1;
                    while j < chunk.len() && (chunk[j] == TOMBSTONE) {
                        j += 1;
                    }

                    if j < chunk.len() {
                        let b = chunk[j];
                        if b != CHUNK_BOUNDARY && b != TOMBSTONE {
                            local_map
                                .entry((a, b))
                                .or_insert_with(Vec::new)
                                .push(base_offset + i);
                        }
                    }
                    i += 1;
                }
                local_map
            })
            .collect();

        // Merge partial maps
        let mut result: AHashMap<(u32, u32), Vec<usize>> = AHashMap::new();
        for partial in partial_maps {
            for (pair, positions) in partial {
                result.entry(pair).or_insert_with(Vec::new).extend(positions);
            }
        }

        result
    }

    /// Find the next valid (non-tombstone, non-boundary) token position after pos
    #[inline]
    fn find_next(&self, tokens: &[u32], pos: usize) -> Option<usize> {
        let mut i = pos + 1;
        while i < tokens.len() {
            if tokens[i] != TOMBSTONE && tokens[i] != CHUNK_BOUNDARY {
                return Some(i);
            }
            if tokens[i] == CHUNK_BOUNDARY {
                return None; // Hit chunk boundary
            }
            i += 1;
        }
        None
    }

    /// Find the previous valid (non-tombstone, non-boundary) token position before pos
    #[inline]
    fn find_prev(&self, tokens: &[u32], pos: usize) -> Option<usize> {
        if pos == 0 {
            return None;
        }
        let mut i = pos - 1;
        loop {
            if tokens[i] != TOMBSTONE && tokens[i] != CHUNK_BOUNDARY {
                return Some(i);
            }
            if tokens[i] == CHUNK_BOUNDARY {
                return None; // Hit chunk boundary
            }
            if i == 0 {
                return None;
            }
            i -= 1;
        }
    }

    /// Merge at given positions using tombstone marking (no array compaction)
    #[allow(clippy::too_many_arguments)]
    fn merge_with_index(
        &self,
        tokens: &mut [u32],
        pair_positions: &mut AHashMap<(u32, u32), Vec<usize>>,
        positions: &[usize],
        pair_a: u32,
        pair_b: u32,
        new_id: u32,
        heap: &mut BinaryHeap<(i64, (u32, u32))>,
    ) {
        if positions.is_empty() {
            return;
        }

        // Collect all index changes in batch (avoid repeated retain calls)
        let mut to_remove: AHashMap<(u32, u32), std::collections::HashSet<usize>> = AHashMap::new();
        let mut to_add: AHashMap<(u32, u32), Vec<usize>> = AHashMap::new();

        // Process each merge position
        for &pos in positions {
            // Skip if already processed (tombstone)
            if tokens[pos] == TOMBSTONE || tokens[pos] != pair_a {
                continue;
            }

            // Find the actual second token position
            let second_pos = match self.find_next(tokens, pos) {
                Some(p) if tokens[p] == pair_b => p,
                _ => continue,
            };

            // Get neighbors before modifying
            let prev_pos = self.find_prev(tokens, pos);
            let next_pos = self.find_next(tokens, second_pos);

            // Batch remove old pairs
            if let Some(pp) = prev_pos {
                let prev_token = tokens[pp];
                to_remove.entry((prev_token, pair_a)).or_default().insert(pp);
            }

            if let Some(np) = next_pos {
                let next_token = tokens[np];
                to_remove.entry((pair_b, next_token)).or_default().insert(second_pos);
            }

            // Apply the merge to tokens
            tokens[pos] = new_id;
            tokens[second_pos] = TOMBSTONE;

            // Batch add new pairs
            if let Some(pp) = prev_pos {
                let prev_token = tokens[pp];
                to_add.entry((prev_token, new_id)).or_default().push(pp);
            }

            if let Some(np) = next_pos {
                let next_token = tokens[np];
                to_add.entry((new_id, next_token)).or_default().push(pos);
            }
        }

        // Apply batch removals (single pass per pair)
        for (pair, remove_set) in to_remove {
            if let Some(idx_positions) = pair_positions.get_mut(&pair) {
                idx_positions.retain(|p| !remove_set.contains(p));
                if idx_positions.is_empty() {
                    pair_positions.remove(&pair);
                }
            }
        }

        // Apply batch additions
        for (pair, add_positions) in to_add {
            let entry = pair_positions.entry(pair).or_insert_with(Vec::new);
            entry.extend(add_positions);
            heap.push((entry.len() as i64, pair));
        }
    }

    fn encode_chunk(&self, chunk: &str) -> Vec<u32> {
        let mut ids: Vec<u32> = chunk.bytes().map(|b| b as u32).collect();

        while ids.len() >= 2 {
            // Find best pair (lowest merge index)
            let mut best_pair: Option<(usize, (u32, u32))> = None;
            let mut best_idx = u32::MAX;

            for i in 0..ids.len() - 1 {
                let pair = (ids[i], ids[i + 1]);
                if let Some(&merge_id) = self.merges.get(&pair) {
                    let idx = merge_id - 256;
                    if idx < best_idx {
                        best_idx = idx;
                        best_pair = Some((i, pair));
                    }
                }
            }

            match best_pair {
                Some((pos, pair)) => {
                    let new_id = self.merges[&pair];
                    ids.splice(pos..pos + 2, std::iter::once(new_id));
                }
                None => break,
            }
        }

        ids
    }
}

/// Python module
#[pymodule]
fn rust_bpe_tokenizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustBPE>()?;
    Ok(())
}
