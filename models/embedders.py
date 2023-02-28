import sentence_transformers


class MiniLM:
    def __init__(self, device="cpu"):
        self.model = sentence_transformers.SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
        self.model.to(device)
        self.model.eval()

    def encode(self, sentences):
        embeddings = self.model.encode(sentences)
        return embeddings
