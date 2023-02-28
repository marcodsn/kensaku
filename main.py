import os
import sys

import faiss

from models import embedders
from tools import index_manager

embedder = embedders.MiniLM("cuda")

# Index generation
idx_gen = False
path = ""
for i, arg in enumerate(sys.argv):
    if arg == "--make_index":
        idx_gen = True
    elif arg == "--path":
        path = sys.argv[i + 1]

if not os.path.exists(path):
    print("A valid path must be provided. Use --path <path>")
    exit()

if idx_gen:
    index_manager.make_index(path, embedder, "indexes/" + path.split("/")[-1] + ".faiss")

# Index loading
index = faiss.read_index("indexes/" + path.split("/")[-1] + ".faiss")

# Query loop
while True:
    query = input("Enter your query: ")
    res = embedder.encode([query])
    print(res.shape)
