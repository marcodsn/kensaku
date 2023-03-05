import os
import sys

import faiss

from models import embedders
from models import llm
from tools import index_manager

embedder = embedders.MiniLM("cuda")
print("Embedder loaded.")

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
print("Index loaded. Articles:", index.ntotal)

# Load files list
with open("indexes/" + path.split("/")[-1] + ".faiss.txt", "r") as f:
    files = f.read().splitlines()
print("Files list loaded. Files:", len(files))

# Model chat loop
model = llm.Pythia()
print("Model loaded.\n")

history = "This is a chat between a human and an AI. The human asks questions and the AI answers them. If the answer is not contained in the retrieved knowledge the AI should answer: \"Sorry but I don\'t have enough data to answer this question.\"\n"
history += "Human: What is the capital of France? EXTERNAL KNOWLEDGE: Paris is the capital and most populous city of France.\n" \
          "AI: Paris is the capital of France.\n"

history += "Human: What is Earth atmosphere composed of? EXTERNAL KNOWLEDGE: The three major constituents of Earth's atmosphere are nitrogen, oxygen, and argon. The atmosphere of Earth is composed of nitrogen (78%), oxygen (21%), argon (0.9%), carbon dioxide (0.04%) and trace gases.\n" \
            "AI: The atmosphere of Earth is composed of nitrogen (78%), oxygen (21%), argon (0.9%), carbon dioxide (0.04%) and trace gases.\n"

while True:
    query = input("You: ")
    res = embedder.encode([query])
    D, I = index.search(res, 2)
    I = [files[i] for i in I[0]]
    knowledge = ""
    # print("Retrieved articles:", I)
    for i in I:
        for sentence in index_manager.get_most_relevant(path + "/" + i, res, embedder, n_results=1):
            # print("Retrieved sentence:", sentence)
            knowledge += sentence + ". "
    print("Retrieved knowledge:", knowledge)
    print("AI:" + model.generate(query, history, knowledge))
