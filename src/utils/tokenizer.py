# Custom wrapper based on GPT2 tokenizer with some extra special
# tokens and helper functions.

import tiktoken as tk
import copy


# Initialize tokenizer
tokenizer = tk.get_encoding('gpt2')

# GPT2 and GPT3 vocabulary size is 50257, rounded to 64-element boundary
# for efficiency.
n_vocab = 50304
_extended_special_tokens = {
    '<|endoftext|>': tokenizer._special_tokens['<|endoftext|>'],  # likely 50256

    # Tokens trained in the chat SFT stage
    '<|user_start|>': 50257,
    '<|user_end|>': 50258,
    '<|assistant_start|>': 50259,
    '<|assistant_end|>': 50260,

    # For evaluating math expressions
    '<|python_start|>': 50261,
    '<|python_end|>': 50262,

    # Should hold the result of the expression with <|python_*|> wrapper.
    '<|output_start|>': 50263,
    '<|output_end|>': 50264
}


def tokenize_conversation(conversation: list):
    '''
    Tokenize a single chat conversation.

    `conversation` should be a list consisting:
    [ { 'content' : 'content string', role: 'either assistant or user' } ... ]

    Returns:
    - tokens: list[int] = A flattened tokenized list
    - mask: list[int] of same length as tokens: mask = 1 (or true), tokens
      which should be accounted during training, exclusively, the assistant
      tokens.
    '''

    res_tokens, res_masks = [], []
    def add_tokens(tokens, mask_value):
        if isinstance(tokens, int):
            tokens = [ tokens ]

        res_tokens.extend(tokens)
        # Luckily for us a series of tokens are masked, instead of single token
        # basis. So a single mask value is enough in this case.
        res_masks.extend([ mask_value ] * len(tokens))

    # From nanochat by Karpathy
    # Merge first system message with following user message
    if conversation[0]['role'] == 'system':
        # some conversation surgery is necessary here for now...
        conversation = copy.deepcopy(conversation) # avoid mutating the original

        assert conversation[1]['role'] == 'user', ('System message must be '
            'followed by a user message')

        # Merge with next user message
        conversation[1]['content'] = (conversation[0]['content'] + '\n\n' +
            conversation[1]['content'])
        conversation = conversation[1:]

    assert len(conversation) >= 1, (f'Conversation has less than 1 '
        f'message: {conversation}')

    # <|endoftext|> can be interpreted as a BOS token in this case.
    bos = _extended_special_tokens['<|endoftext|>']
    user_start = _extended_special_tokens['<|user_start|>']
    user_end = _extended_special_tokens['<|user_end|>']
    assistant_start = _extended_special_tokens['<|assistant_start|>']
    assistant_end = _extended_special_tokens['<|assistant_end|>']
    python_start = _extended_special_tokens['<|python_start|>']
    python_end = _extended_special_tokens['<|python_end|>']
    output_start = _extended_special_tokens['<|output_start|>']
    output_end = _extended_special_tokens['<|output_end|>']

    add_tokens(bos, 0)
    for i, message in enumerate(conversation):
        # Sanity check
        role = message['role']
        must_be_from = 'user' if i % 2 == 0 else 'assistant'
        assert role == must_be_from, (f'Message {i} must be form '
            f'{must_be_from}, found {role}!')

        # Content can either be plain string or a list of parts containing
        # tool calls, such as python expression evaluation
        content = message['content']

        if role == 'user':
            add_tokens(user_start, 0)
            add_tokens(tokenizer.encode_ordinary(content), 0)
            add_tokens(user_end, 0)
        elif role == 'assistant':
            add_tokens(assistant_start, 0)

            # String content, simply encode and append.
            if isinstance(content, str):
                add_tokens(tokenizer.encode_ordinary(content), 1)

            # If content is a list of parts
            elif isinstance(content, list):
                for part in content:
                    # Every part contains text
                    value_tokens = tokenizer.encode_ordinary(part['text'])

                    # Just plain text
                    if part['type'] == 'text':
                        add_tokens(value_tokens, 1)

                    # Mostly mathematical expression that should be evaluated
                    # using Python.
                    elif part['type'] == 'python':
                        add_tokens(python_start, 1)
                        add_tokens(value_tokens, 1)
                        add_tokens(python_end, 1)

                    # Mathematical expression result, python output
                    elif part['type'] == 'output':
                            add_tokens(output_start, 0)
                            add_tokens(value_tokens, 0)
                            add_tokens(output_end, 0)
                    else:
                        raise ValueError(f'Unknown part type: {part['type']}')
            else:
                raise ValueError(f'Unknown content type: {type(content)}')

            # End of assistant generation
            add_tokens(assistant_end, 1)

    return res_tokens, res_masks
