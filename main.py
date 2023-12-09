from transformers import AutoTokenizer

# Constants
stdout_padding = "#" * 20

# Load the tokenizer
tokenizer_path = "/home/bradley/data/llama-2-tokenizer"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Confirm vocabulary size
print(f"{stdout_padding} Llama2 Tokenizer Details {stdout_padding}\n")
print(f"Llama2 tokenizer overview: {tokenizer}")
print(f"Llama2 Vocabulary Size: {len(tokenizer.get_vocab().keys())}\n")
print(f"{stdout_padding} End of Llama2 Tokenizer Details {stdout_padding}\n")

# Verify token IDs for Llama2 special tokens
print(f"{stdout_padding} Llama2 Special Tokens {stdout_padding}\n")

UNK = "<unk>" # Unknown token
BOS, EOS = "<s>", "</s>" # Begin of sequnece and end of sequence tokens

special_tokens = [UNK, BOS, EOS]

for token in special_tokens:
    print(f'Token ID for the special token {token}: {tokenizer.get_vocab()[token]}')
    print(f'Encoded {token} becomes: {tokenizer.encode(token)}\n')

print(f"{stdout_padding} End of Llama2 Special Tokens {stdout_padding}\n")

# Verify token IDs for Llama2 prompt symbols
print(f"{stdout_padding} Llama2 Prompt Symbols {stdout_padding}\n")

B_INST, E_INST = "[INST]", "[/INST]" # Begin of instruction and end of instruction symbols
B_SYS, E_SYS = "<<SYS>>\n", "\n<<SYS>>\n\n" # Begin of system message and end of system message symbols

prompt_symbols = [B_INST, E_INST, B_SYS, E_SYS]

for symbol in prompt_symbols:
    encoded_symbol = tokenizer.encode(symbol)
    print(f'Encoded {repr(symbol)} becomes: {encoded_symbol}')
    
    for token in encoded_symbol:
        print(f"\tToken ID {token} --> {repr(tokenizer.decode(token))}")

print(f"\n{stdout_padding} End of Llama2 Prompt Symbols {stdout_padding}\n")

# Test tokenizer on a sentence
print(f"{stdout_padding} Llama2 Tokenizer Sentence Example {stdout_padding}\n")
sentence = "RHEL subscription manager let's you manage packages on RedHat."
encoded_output = tokenizer.encode(sentence)
print(f"Original sentence: {sentence}")
print(f"Encoded sentence: {encoded_output}")

# Verify what each token ID correlates to
for token in encoded_output:
    print(f"Token ID {token} --> {tokenizer.decode(token)}")

print(f"{stdout_padding} End of Llama2 Tokenizer Sentence Example {stdout_padding}\n")