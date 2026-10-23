import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
from interpretable_extractive_graphs.utils.document import Document
from interpretable_extractive_graphs.utils.text_embedder import TextEmbedder


@pytest.fixture(scope="session")
def embedder():
    return TextEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")


@pytest.fixture
def sentences():
    return [
        "Apple released a new product today.",
        "The stock market closed higher on Tuesday.",
        "A cat slept on the warm windowsill.",
    ]


class TestTextEmbedderInit:
    def test_init_stores_config(self, embedder):
        assert embedder.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert embedder.batch_size == 32
        assert embedder.device in ("cpu", "cuda")

    def test_init_accepts_batch_size_and_device(self):
        e = TextEmbedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=8,
            device="cpu",
        )
        assert e.batch_size == 8
        assert e.device == "cpu"

    def test_init_invalid_model_raises(self):
        with pytest.raises(ValueError, match="not found"):
            TextEmbedder(model_name="definitely/not-a-real-model-xyz")


class TestDeviceResolution:
    def test_auto_detect_returns_valid_device(self):
        e = TextEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
        assert e.device in ("cpu", "cuda")

    def test_auto_detect_matches_torch(self):
        import torch

        e = TextEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        assert e.device == expected

    def test_cpu_forced_places_model_on_cpu(self):
        e = TextEmbedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
        )
        assert e.device == "cpu"
        assert e.model.device.type == "cpu"
        assert e.is_gpu() is False

    def test_invalid_device_raises(self):
        with pytest.raises(ValueError, match="Invalid device"):
            TextEmbedder(
                model_name="sentence-transformers/all-MiniLM-L6-v2", device="gpu"
            )

    def test_cuda_request_without_gpu_raises(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(RuntimeError, match="CUDA is not available"):
            TextEmbedder(
                model_name="sentence-transformers/all-MiniLM-L6-v2", device="cuda"
            )

    def test_cuda_no_silent_fallback(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(RuntimeError):
            TextEmbedder(
                model_name="sentence-transformers/all-MiniLM-L6-v2", device="cuda"
            )
        # auto-detect must still yield cpu, not raise
        e = TextEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
        assert e.device == "cpu"


class TestEmbedSentences:
    def test_shape_is_2d_and_matches_input(self, embedder, sentences):
        result = embedder.embed_sentences(sentences)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 384)

    def test_single_sentence_returns_2d(self, embedder):
        result = embedder.embed_sentences(["A single sentence here."])
        assert result.ndim == 2
        assert result.shape == (1, 384)

    def test_dtype_is_float32(self, embedder, sentences):
        assert embedder.embed_sentences(sentences).dtype == np.float32

    def test_vectors_are_unit_normalized(self, embedder, sentences):
        result = embedder.embed_sentences(sentences)
        norms = np.linalg.norm(result, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_dot_product_equals_cosine_similarity(self, embedder, sentences):
        result = embedder.embed_sentences(sentences)
        manual = sentences[0] and (
            embedder.model.encode(sentences[0], normalize_embeddings=True)
            @ result[1]
        )
        recomputed = result[0] @ result[1]
        assert np.isclose(manual, recomputed, atol=1e-5)

    def test_bare_string_raises_type_error(self, embedder):
        with pytest.raises(TypeError, match="list\\[str\\]"):
            embedder.embed_sentences("A single string, not a list.")

    def test_empty_list_raises(self, embedder):
        with pytest.raises(ValueError, match="empty sentence list"):
            embedder.embed_sentences([])

    def test_whitespace_sentence_raises_with_index(self, embedder):
        bad = ["Valid sentence.", "   ", "Another valid sentence."]
        with pytest.raises(ValueError, match="index 1"):
            embedder.embed_sentences(bad)

    def test_over_long_sentence_raises(self, embedder):
        long_sentence = " ".join(["word"] * (embedder.model.max_seq_length + 50))
        with pytest.raises(ValueError, match="model limit"):
            embedder.embed_sentences([long_sentence])

    def test_error_message_has_space_before_next_sentence(self, embedder):
        bad = ["Valid sentence.", "", "Another valid sentence."]
        with pytest.raises(ValueError) as exc_info:
            embedder.embed_sentences(bad)
        message = str(exc_info.value)
        assert ".Check" not in message
        assert ". Run" not in message


class TestEmbedDocument:
    def test_sets_embeddings_and_returns_same_doc(self, embedder, sentences):
        doc = Document(raw_text="Some raw text.")
        doc.sentences = sentences
        result = embedder.embed_document(doc)
        assert result is doc
        assert doc.embeddings is not None
        assert doc.embeddings.shape == (3, 384)

    def test_embeddings_align_with_sentence_order(self, embedder, sentences):
        doc = Document(raw_text="Some raw text.")
        doc.sentences = sentences
        embedder.embed_document(doc)

        expected = embedder.embed_sentences(sentences)
        assert np.allclose(doc.embeddings[2], expected[2], atol=1e-6)

    def test_preserves_case_in_source_text(self, embedder, sentences):
        doc = Document(raw_text="Some raw text.")
        doc.sentences = sentences
        embedder.embed_document(doc)
        assert doc.sentences[0].startswith("Apple")

    def test_no_sentences_raises(self, embedder):
        doc = Document(raw_text="Raw text with no sentences processed yet.")
        with pytest.raises(ValueError, match="Preprocessor"):
            embedder.embed_document(doc)