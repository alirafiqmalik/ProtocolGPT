import os
import tiktoken
from utils import get_all_files
from consts import LOADER_MAPPING

enc = tiktoken.get_encoding("cl100k_base")
assert enc.decode(enc.encode("hello world")) == "hello world"

# To get the tokeniser corresponding to a specific model in the OpenAI API:
enc = tiktoken.encoding_for_model("gpt-4")

def num_tokens_from_string(string: str, model_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def num_tokens_from_directory(directory: str, model_name: str) -> int:
    """Returns the number of tokens in a directory of text files."""
    total_tokens = 0
    files = get_all_files(directory)
    for filename in files:
        for ext in LOADER_MAPPING:
            if filename.endswith(ext):
                print(filename)
                with open(os.path.join(directory, filename), "r") as f:
                    text = f.read()
                    num_tokens = num_tokens_from_string(text, model_name)
                    total_tokens += num_tokens
    return total_tokens

if __name__=="__main__":
    # print(num_tokens_from_string("www com www 123", "gpt-3.5-turbo"))
    print(num_tokens_from_directory("/home/why/sec_sea/Fuzzers/ProtocolGPT/ProtocolGPT/test_code/orig_repo/feng", "gpt-3.5-turbo"))