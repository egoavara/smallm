# CLAUDE.md

이 파일은 Claude Code (claude.ai/code)가 이 저장소의 코드 작업 시 참고하는 가이드입니다.

## 프로젝트 개요

SmallM은 커스텀 BPE 토크나이저(Rust/Python)를 포함한 교육용 미니멀 LLaMA 스타일 언어 모델 구현체입니다. Hugging Face: https://huggingface.co/egoavara/smallm

## 빌드 명령어

### Python 환경
```bash
uv sync  # 의존성 설치
```

### Rust BPE 토크나이저 (선택사항 - 더 빠른 토큰화)
```bash
cd rust-bpe-tokenizer && \
  unset CONDA_PREFIX && \
  PATH="../.venv/bin:$PATH" VIRTUAL_ENV="../.venv" \
  maturin develop --release
```

## 아키텍처

### 핵심 모듈
```
src/smallm/
├── model/       # LLaMA 트랜스포머 (RoPE, GQA, SwiGLU FFN, RMSNorm)
├── tokenizer/   # BPE 구현체 (Rust 래퍼, Numba 최적화 Python)
├── data/        # 스트리밍 데이터셋 로딩 (wikitext, tinystories, openwebtext, 인스트럭션 데이터셋)
├── training/    # 학습 루프, 체크포인트 관리, Jupyter UI
└── quantization/# 학습 후 양자화 (FP16, INT8)
```

### 진입점
- `train-tokenizer.py` - 데이터셋 샘플로 BPE 토크나이저 학습
- `train-model.py` - Jupyter UI 컨트롤로 LLaMA 모델 학습
- `test-model.py` - 모델 검사 및 설정
- `test-chat.py` - 대화형 채팅 인터페이스
- `upload-huggingface.py` - HF Hub에 모델 업로드/다운로드

### 설정 시스템 (`config.py`)
`build/config.json`에 저장되는 2단계 설정:
- **모드**: `base` (wikitext, tinystories, openwebtext) 또는 `instruct` (openassistant, alpaca, dolly, ultrachat)
- **모델 크기**: `tiny`, `small`, `medium`, `large` (레이어 수, 헤드 수, d_model에 영향)

토크나이저 선택:
```python
config.tokenizer.bpe_type = "rust"   # 빠름, Rust 빌드 필요
config.tokenizer.bpe_type = "python" # Numba JIT 폴백
```

### 주요 설계 패턴
- **가중치 공유**: 토큰 임베딩과 출력 프로젝션 공유
- **Pre-norm**: 서브레이어 이전에 RMSNorm 적용 (이후가 아님)
- **GQA**: 공유 KV 헤드로 추론 시 메모리 절감
- **RoPE**: 회전 위치 임베딩 (학습된 위치 임베딩 없음)
- **ChatML 형식**: 대화에 `<|im_start|>role\ncontent<|im_end|>` 사용
- **스트리밍 데이터셋**: 모든 데이터셋이 `streaming=True`로 로드

### 체크포인트 관리
체크포인트 저장 위치: `build/{mode}/{model_size}/checkpoints/`
- 형식: `step_XXXXXX_loss_Y.YYYY.pt`
- 최고 체크포인트는 `best.pt`로 심볼릭 링크
- 손실 이력은 `loss_history.json`에 기록
