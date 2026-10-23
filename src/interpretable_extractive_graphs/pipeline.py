import numpy as np
from interpretable_extractive_graphs.utils.document import Document
from interpretable_extractive_graphs.utils.preprocessor import Preprocessor
from interpretable_extractive_graphs.utils.text_embedder import TextEmbedder
from interpretable_extractive_graphs.utils.graph_builder import GraphBuilder

text = "Text summarization is an important area of natural language processing (NLP) that focuses on condensing large amounts of text into shorter, coherent summaries. Modern approaches can identify the main ideas in a document and present them with minimal human involvement. Extractive methods select representative sentences directly from the source text, while abstractive methods generate new phrasing based on the original meaning."
model_name = "sentence-transformers/all-MiniLM-L6-v2"
batch_size = 32
device = "cpu"

doc = Document(raw_text=text)
print(doc.raw_text)
preprocessor = Preprocessor()

doc = preprocessor.process(doc)
# doc.cleaned_text = preprocessor.clean(doc)
print("cleaned text: ", doc.cleaned_text)
# doc.sentences = preprocessor.split_sentence(doc)
print("splited text: ", doc.sentences)

embedder = TextEmbedder(model_name=model_name, batch_size=batch_size, device=device)
embedded_sent = embedder.embed_sentences(sentences=doc.sentences)
print(embedded_sent)

graph = GraphBuilder()
similarity = graph.compute_similarity(embedded_sent)
print("Similarity matrix: ", similarity)

print("Matrix is square") if similarity.ndim == 2 and similarity.shape[
    0
] == similarity.shape[1] else print("Matrix is not square")

print("Matrix has zero diagonal") if np.all(np.diag(similarity) == 0.0) else print(
    "Matris has no zero diagonal"
)

print("Matrix is all positive") if np.all(similarity >= 0.0) else print(
    "Matrix has negatives"
)

print("Matrix is symmetric") if np.allclose(similarity, similarity.T) else print(
    "Matrix is not symmetric"
)

graph.add_edge_similarity(similarity, threshold=0.25)

di_graph = GraphBuilder(directed=True)

parsed_dep = preprocessor.parse_dependency(doc)
