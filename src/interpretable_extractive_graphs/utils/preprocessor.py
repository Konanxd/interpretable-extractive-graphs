import string
import spacy
from spacy.language import Language
from interpretable_extractive_graphs.utils.document import Document

# Dependency labels that carry topical/semantic signal between sentences.
SEMANTIC_LABELS = frozenset(
    {
        "nsubj",
        "nsubjpass",
        "dobj",
        "pobj",
        "attr",
        "oprd",
        "ROOT",
        "ccomp",
        "xcomp",
        "advcl",
        "appos",
    }
)

# Discourse cue words -> relation type. Weights follow the experiment plan:
# Contrast=1.0, Cause-Result=0.8, Elaboration=0.6.
DISCOURSE_CUES = {
    "contrast": (
        "however",
        "but",
        "although",
        "though",
        "whereas",
        "despite",
        "nevertheless",
        "nonetheless",
        "instead",
        "yet",
    ),
    "cause": ("therefore", "thus", "consequently", "because", "since", "hence"),
    "elaboration": (
        "moreover",
        "furthermore",
        "additionally",
        "specifically",
        "particularly",
        "indeed",
    ),
    "sequence": ("first", "then", "next", "finally", "subsequently", "afterwards"),
}

CUE_WEIGHTS = {
    "contrast": 1.0,
    "cause": 0.8,
    "elaboration": 0.6,
    "sequence": 0.6,
}


class Preprocessor:
    def __init__(
        self,
        lowercase: bool = False,
        remove_punctuations: bool = False,
        disable_pipes: tuple[str, ...] | None = None,
    ):
        self.lowercase = lowercase
        self.remove_punctuations = remove_punctuations

        load_kwargs = {} if disable_pipes is None else {"disable": disable_pipes}
        self.nlp: Language = spacy.load("en_core_web_sm", **load_kwargs)

        if not self.nlp.has_pipe("parser"):
            raise RuntimeError(
                "en_core_web_sm must have its dependency parser enabled."
            )

    def clean(self, doc: Document) -> Document:
        text = doc.raw_text

        if self.lowercase:
            text = text.lower()
        if self.remove_punctuations:
            text = text.translate(str.maketrans("", "", string.punctuation))

        doc.cleaned_text = text.strip()
        return doc

    def split_sentence(self, doc: Document) -> Document:
        parsed = self.nlp(doc.raw_text)

        doc.sentences = [sentence.text.strip() for sentence in parsed.sents]

        return doc

    def parse_dependency(self, doc: Document) -> Document:
        """Extract dependency arcs, semantic role heads, and discourse cues."""
        parsed = self.nlp(doc.raw_text)

        dep_arcs: list[list[tuple[str, str, str]]] = []
        role_heads: list[list[str]] = []
        discourse_cues: list[list[tuple[str, str]]] = []

        for sentence in parsed.sents:
            arcs: list[tuple[str, str, str]] = []
            heads: list[str] = []
            cues: list[str] = []

            for token in sentence:
                arcs.append((token.head.text, token.dep_, token.text))

                if token.dep_ in SEMANTIC_LABELS and token.pos_ != "PUNCT":
                    token_lower = token.text.lower()

                    if token_lower not in heads:
                        heads.append(token_lower)

                token_lower = token.text.lower()

                for relation, cue_words in DISCOURSE_CUES.items():
                    if token_lower in cue_words:
                        cues.append((token_lower, relation))
                        break

            dep_arcs.append(arcs)
            role_heads.append(heads)
            discourse_cues.append(cues)

        doc.dep_arcs = dep_arcs
        doc.role_heads = role_heads
        doc.discourse_cues = discourse_cues

        return doc

    def process(self, doc: Document) -> Document:
        """Run full preprocessing pipeline in correct order."""
        self.clean(doc)
        self.split_sentence(doc)
        return doc
