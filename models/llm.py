from transformers import GPTNeoXForCausalLM, AutoTokenizer


class Pythia:
    def __init__(self):
        self.model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/pythia-6.9b-deduped",
                                                        device_map="auto",
                                                        load_in_8bit=True)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-6.9b-deduped")

    def generate(self, query, history, knowledge, max_new_tokens=32, search_strategy=None):
        prompt = history + "Human: " + query + "EXTERNAL_KNOWLEDGE: " + knowledge + "\n" + "AI:"
        inputs = self.tokenizer(prompt, return_tensors="pt").to('cuda')
        tokens = self.model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=0)
        answer = self.tokenizer.decode(tokens[0])[len(prompt):].split("\n")[0]
        return answer
