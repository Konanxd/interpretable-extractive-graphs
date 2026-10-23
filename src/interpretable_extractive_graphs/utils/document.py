from typing import Any, Optional
import numpy as np


class Document:
    def __init__(self, raw_text: str, metadata: Optional[dict[str, Any]] = None):
        if not raw_text or not raw_text.strip():
            raise ValueError("raw_text cannot be empty.")
        self.raw_text: str = raw_text
        self.cleaned_text: str = ""
        self.sentences: list[str] = []
        self.metadata: dict[str, Any] = metadata or {}
        self.tokens: list[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.predictions: dict[str, Any] = {}

        self.dep_arcs: list[list[tuple[str, str, str]]] = []
        self.role_heads: list[list[str]] = []
        self.discourse_cues: list[list[tuple[str, str]]] = []

    def __repr__(self):
        return f"Document(sentences={len(self.sentences)}, "
