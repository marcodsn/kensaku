# Kensaku

Kensaku is a command-line tool for answering questions based on text data. It uses pre-trained language models and FAISS
to retrieve relevant sentences and generate answers to questions.

## Installation

Clone the Kensaku repository:

```git clone https://github.com/marcodsn/kensaku.git```

Install the required packages:

```pip install -r requirements.txt```

## Usage

Kensaku can be used to answer questions based on text data. It uses a pre-trained MiniLM language model and FAISS to
retrieve relevant sentences and generate answers to questions.

**NOTE**: this tool was tested using a Wikipedia dataset where each article has its own JSON file and a **'text'**
section within it. Here is an example of file structure:

```
{
"title": "Earth",
"text": "The Earth is the third planet..."
}
```

If your dataset has a different format you will need to modify the code accordingly.

### Generating an index

Before you can use Kensaku to answer questions, you need to generate an index of the text data you want to search. You
can generate an index by running Kensaku with the **--make_index** flag and providing a path to the directory containing
the
text data you want to index:

```python main.py --make_index --path /path/to/text/data```

This will generate an index file and a list of files (this is just a workaround, will be removed ASAP) in the **'
indexes'** directory.

### Asking questions

Once you have generated an index, you can use Kensaku to answer questions. To ask a question, simply run Kensaku without
the --make_index flag:

```python main.py --path /path/to/text/data```

Kensaku will prompt you to enter a question. It will then retrieve the most relevant sentences from the indexed text
data and use them to generate an answer to the question.

## Example

Here's an example of how Kensaku can be used to answer questions:

```python main.py --make_index --path data```

```bash
$ python main.py --path data
Embedder loaded.
Index loaded. Articles: 6281203
Files list loaded. Files: 6281203
Model loaded.

You: What is the capital of France?
Retrieved knowledge: Paris is the capital and most populous city of France.
AI: Paris is the capital of France.

You: What is Earth atmosphere composed of?
Retrieved knowledge: The three major constituents of Earth's atmosphere are nitrogen, oxygen, and argon. The atmosphere
of Earth is composed of nitrogen (78%), oxygen (21%), argon (0.9%), carbon dioxide (0.04%) and trace gases.
AI: The atmosphere of Earth is composed of nitrogen (78%), oxygen (21%), argon (0.9%), carbon dioxide (0.04%) and trace
gases.
```

## FAQ

- **Why is Kensaku so slow?** Currently there are a lot of possible optimizations to be made, like using a different
  FAISS index, removing the need to load the list of files, and more. I will be working on this in the future.
- **Why is Kensaku so inaccurate?** Kensaku is still in its early stages of development and is not very accurate. Take
  into account that it is just an experimental project, and it is not meant to be used in production.
- **What are the VRAM requirements for Kensaku?** At the moment, Kensaku requires just over 8GB of VRAM to run the
  embedding model and the LLM model to generate answers. This can be reduced by changing the LLM model to a smaller one
  manually (in models/llm.py), check out https://huggingface.co/EleutherAI/pythia-6.9b-deduped to find other available sizes
  for the Pythia model (otherwise you can use a completely different LLM too!).

## License

Kensaku is licensed under the MIT License. See the LICENSE file for more information.