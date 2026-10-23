import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from interpretable_extractive_graphs.utils.document import Document


class TestDocument:
    def test_init_with_valid_text(self):
        doc = Document(raw_text="Hello world. This is a test.")
        assert doc.raw_text == "Hello world. This is a test."
        assert doc.sentences == []
        assert doc.metadata == {}

    def test_init_with_metadata(self):
        meta = {"source": "test", "id": 1}
        doc = Document(raw_text="Sample text.", metadata=meta)
        assert doc.metadata["source"] == "test"
        assert doc.metadata["id"] == 1

    def test_init_empty_text_raises(self):
        with pytest.raises(ValueError):
            Document(raw_text="")

    def test_init_whitespace_only_raises(self):
        with pytest.raises(ValueError):
            Document(raw_text="   \n\t  ")

    def test_repr_returns_string(self):
        doc = Document(raw_text="Test.")
        result = repr(doc)
        assert isinstance(result, str)
        assert "Document" in result

    def test_default_values(self):
        doc = Document(raw_text="Test.")
        assert doc.cleaned_text == ""
        assert doc.tokens == []
        assert doc.embeddings is None
        assert doc.predictions == {}

    def test_dependency_fields_default_empty(self):
        doc = Document(raw_text="Test.")
        assert doc.dep_arcs == []
        assert doc.role_heads == []
        assert doc.discourse_cues == []
