import numpy as np
import torch
from typing import Optional
from sentence_transformers import SentenceTransformer

from interpretable_extractive_graphs.utils.document import Document

VALID_DEVICES = ("cpu", "cuda")


class TextEmbedder:
    def __init__(
        self, model_name: str, batch_size: int = 32, device: Optional[str] = None
    ):
        self.device = self._resolve_device(device)

        try:
            self.model = SentenceTransformer(model_name, device=self.device)
        except (OSError, ValueError) as e:
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Check the model name or your internet connection."
            ) from e

        self.model_name = model_name
        self.batch_size = batch_size

    def _resolve_device(self, device: Optional[str]) -> str:
        if device is None:
            return "cuda" if torch.cuda.is_available() else "cpu"

        if device not in VALID_DEVICES:
            raise ValueError(
                f"Invalid device '{device}'. Valid options: {VALID_DEVICES}."
            )

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "device='cuda' was requested but CUDA is not available. "
                "Attach a GPU runtime, or pass device='cpu' explicitly."
            )

        return device

    def is_gpu(self) -> bool:
        return self.model.device.type == "cuda"

    def _assert_no_truncation(self, sentences: list[str]) -> None:
        max_allowed = self.model.max_seq_length
        if max_allowed is None:
            return

        raw = self.model.tokenizer(sentences, truncation=False, padding=False)[
            "input_ids"
        ]
        longest = max(len(ids) for ids in raw)

        if longest > max_allowed:
            raise ValueError(
                f"Longest sentence is {longest} tokens, model limit is "
                f"{max_allowed}. Truncation would silently drop content."
            )

    def embed_sentences(self, sentences: list[str]) -> np.ndarray:
        if isinstance(sentences, str):
            raise TypeError(
                "embed_sentences expects a list[str], got a single string. "
                "Wrap it in a list or use embed_document()."
            )

        if not sentences:
            raise ValueError("Cannot embed an empty sentence list.")

        for i, sentence in enumerate(sentences):
            if not sentence or not sentence.strip():
                raise ValueError(
                    f"Sentence at index {i} is empty or whitespace-only. "
                    "Check Preprocessor.split_sentence output before embedding."
                )

        self._assert_no_truncation(sentences)

        embeddings = self.model.encode(
            sentences,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        embeddings = np.asarray(embeddings, dtype=np.float32)
        assert embeddings.shape[0] == len(sentences)

        return embeddings

    def embed_document(self, doc: Document) -> Document:
        if not doc.sentences:
            raise ValueError(
                "Document has no sentences. Run Preprocessor.process() first."
            )

        doc.embeddings = self.embed_sentences(doc.sentences)
        return doc
