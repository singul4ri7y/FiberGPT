# Custom wrapper based on GPT2 tokenizer, but with some extra special
# tokens and helper functions.

import tiktoken as tk


# Initialize tokenizer
tokenizer = tk.get_encoding('gpt2')

# GPT2 and GPT3 vocabulary size is 50257, rounded to 64-element boundary
# for efficiency.
n_vocab = 50304
_extended_special_tokens = {
    '<|endoftext|>': tokenizer._special_tokens['<|endoftext|>'],
    '<|user_start|>': 50257,
    '<|user_end|>': 50258,
    '<|assistant_start|>': 50259,
    '<|assistant_end|>': 50260
}
