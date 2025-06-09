import sys
import os

import tiktoken
from git import Repo
from langchain.vectorstores import FAISS
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from consts import LOADER_MAPPING, EXCLUDE_FILES

# todo
def get_repo():
    try:
        return Repo()
    except:
        return None


class StreamStdOut(StreamingStdOutCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        sys.stdout.write(token)
        sys.stdout.flush()

    def on_llm_start(self, serialized, prompts, **kwargs):
        sys.stdout.write("🤖 ")

    def on_llm_end(self, response, **kwargs):
        sys.stdout.write("\n")
        sys.stdout.flush()




def get_all_files(folder):
    file_paths = []
    for root, directories, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths

# load all files in local dir
# todo: add LanguageParser in Loader
def load_local_files(folder):
    files = []
    files_paths = get_all_files(folder)
    for file in files_paths:
        if any(file.endswith(exclude_file) for exclude_file in EXCLUDE_FILES):
            continue
        for ext in LOADER_MAPPING:
            if file.endswith(ext):
                print('\r' + f'📂 Loading files: {file}')
                args = LOADER_MAPPING[ext]['args']
                loader = LOADER_MAPPING[ext]['loader'](file, *args)
                files.extend(loader.load())
    return files


# load files in git repo
def load_git_files():
    repo = get_repo()
    if repo is None:
        return []
    files = []
    tree = repo.tree()
    for blob in tree.traverse():
        path = blob.path
        if any(
                path.endswith(exclude_file) for exclude_file in EXCLUDE_FILES):
            continue
        for ext in LOADER_MAPPING:
            if path.endswith(ext):
                print('\r' + f'📂 Loading files: {path}')
                args = LOADER_MAPPING[ext]['args']
                loader = LOADER_MAPPING[ext]['loader'](path, *args)
                files.extend(loader.load())
    return files


def calculate_cost(texts, model_name):
    enc = tiktoken.encoding_for_model(model_name)
    all_text = ''.join([text.page_content for text in texts])
    tokens = enc.encode(all_text)
    token_count = len(tokens)
    cost = (token_count / 1000) * 0.0004
    return cost


def get_local_vector_store(embeddings, path):
    try:
        return FAISS.load_local(path, embeddings)
    except:
        return None
