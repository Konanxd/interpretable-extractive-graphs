import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from interpretable_extractive_graphs.utils.document import Document
from interpretable_extractive_graphs.utils.preprocessor import Preprocessor


@pytest.fixture
def preprocessor():
    return Preprocessor(lowercase=False, remove_punctuations=False)


@pytest.fixture
def sample_doc():
    return Document(raw_text="Hello world. This is a test. Another sentence!")


@pytest.fixture
def discourse_doc():
    text = (
        "The central bank raised rates because inflation climbed sharply. "
        "However, analysts disagreed. Moreover, growth slowed. "
        "Therefore, the central bank reversed course."
    )
    return Document(raw_text=text)


class TestPreprocessor:
    def test_clean_preserves_case(self, preprocessor, sample_doc):
        preprocessor.clean(sample_doc)
        assert (
            sample_doc.cleaned_text == "Hello world. This is a test. Another sentence!"
        )

    def test_clean_lowercases_when_enabled(self, sample_doc):
        prep = Preprocessor(lowercase=True)
        prep.clean(sample_doc)
        assert (
            sample_doc.cleaned_text == "hello world. this is a test. another sentence!"
        )

    def test_split_sentence_count(self, preprocessor, sample_doc):
        preprocessor.split_sentence(sample_doc)
        assert len(sample_doc.sentences) == 3

    def test_split_sentence_content(self, preprocessor, sample_doc):
        preprocessor.split_sentence(sample_doc)
        assert sample_doc.sentences[0] == "Hello world."
        assert sample_doc.sentences[1] == "This is a test."

    def test_parse_dependency_produces_aligned_lists(
        self, preprocessor, discourse_doc
    ):
        preprocessor.split_sentence(discourse_doc)
        preprocessor.parse_dependency(discourse_doc)
        n = len(discourse_doc.sentences)
        assert len(discourse_doc.dep_arcs) == n
        assert len(discourse_doc.role_heads) == n
        assert len(discourse_doc.discourse_cues) == n

    def test_role_heads_contain_no_punctuation(self, preprocessor, discourse_doc):
        preprocessor.parse_dependency(discourse_doc)
        for heads in discourse_doc.role_heads:
            for head in heads:
                assert head != "."
                assert head != ","

    def test_role_heads_are_unique_and_lowercase(self, preprocessor, discourse_doc):
        preprocessor.parse_dependency(discourse_doc)
        for heads in discourse_doc.role_heads:
            assert len(heads) == len(set(heads))
            for head in heads:
                assert head == head.lower()

    def test_discourse_cues_detected(self, preprocessor, discourse_doc):
        preprocessor.parse_dependency(discourse_doc)
        cues = discourse_doc.discourse_cues
        assert ("because", "cause") in cues[0]
        assert ("however", "contrast") in cues[1]
        assert ("moreover", "elaboration") in cues[2]
        assert ("therefore", "cause") in cues[3]

    def test_dep_arcs_are_head_label_dep_triples(self, preprocessor, discourse_doc):
        preprocessor.parse_dependency(discourse_doc)
        for arcs in discourse_doc.dep_arcs:
            for arc in arcs:
                assert len(arc) == 3
                assert all(isinstance(part, str) for part in arc)

    def test_disable_pipes_skips_unused_components(self):
        prep = Preprocessor(disable_pipes=("ner", "lemmatizer"))
        assert "ner" not in prep.nlp.pipe_names
        assert "lemmatizer" not in prep.nlp.pipe_names
        assert "parser" in prep.nlp.pipe_names

    def test_default_loads_all_pipes(self, preprocessor):
        assert preprocessor.nlp.pipe_names == [
            "tok2vec",
            "tagger",
            "parser",
            "attribute_ruler",
            "lemmatizer",
            "ner",
        ]

    def test_process_runs_all_stages(self, preprocessor, discourse_doc):
        result = preprocessor.process(discourse_doc)
        assert result is discourse_doc
        assert len(result.sentences) == 4
        assert len(result.dep_arcs) == 4
        assert len(result.role_heads) == 4
        assert len(result.discourse_cues) == 4