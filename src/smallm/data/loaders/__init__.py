"""Dataset loaders.

이 패키지를 import하면 모든 로더가 자동으로 registry에 등록됩니다.
"""

from .wikitext import load_wikitext_dataset
from .tinystories import load_tinystories_dataset
from .openwebtext import load_openwebtext_dataset

__all__ = [
    "load_wikitext_dataset",
    "load_tinystories_dataset",
    "load_openwebtext_dataset",
]
