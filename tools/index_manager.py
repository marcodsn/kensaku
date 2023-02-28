import json
import os
import time

import faiss
import numpy as np


def make_index(path, embedder, index_name, batch_size=100000):
    index = faiss.IndexFlatL2(embedder().encode([""]).shape[1])
    files = [f for f in os.listdir(path) if f.endswith(".json")]

    for i in range(0, len(files), batch_size):
        embeddings = np.empty((0, 384), dtype=np.float32)
        texts = []
        st = time.time()
        print("Indexing", i, "to", i + batch_size, "articles...")
        for file in files[i:i + batch_size]:
            with open(path + file, "r") as f:
                data = json.load(f)
                texts.append(data["text"])
        embeddings = np.concatenate((embeddings, embedder.encode(texts)))
        index.add(embeddings)
        et = time.time()
        print("Indexed", i * batch_size + batch_size, "articles.", "Bath time:", et - st)

    faiss.write_index(index, index_name)
    print("Index saved.")
