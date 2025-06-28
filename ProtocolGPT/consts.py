# consts.py

import os
from pathlib import Path

from langchain.document_loaders import CSVLoader, UnstructuredWordDocumentLoader, UnstructuredEPubLoader, PDFMinerLoader, UnstructuredMarkdownLoader, TextLoader

# api key of hhh
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "****")
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1" # Add this line
EXCLUDE_DIRS = ['__pycache__', '.venv', '.git', '.idea', 'venv', 'env', 'node_modules', 'dist', 'build', '.vscode',
                '.github', '.gitlab']
ALLOW_FILES = ['.txt', '.js', '.mjs', '.ts', '.tsx', '.css', '.scss', '.less', '.html', '.htm', '.json', '.py',
               '.java', '.c', '.cpp', '.cs', '.go', '.php', '.rb', '.rs', '.swift', '.kt', '.scala', '.m', '.h',
               '.sh', '.pl', '.pm', '.lua', '.sql']
EXCLUDE_FILES = ['requirements.txt', 'package.json', 'package-lock.json', 'yarn.lock']
MODEL_TYPES = {
    "OPENAI": "openai",
    "LOCAL": "local",
    "OPENROUTER": "openrouter", # Add this line
}
DEFAULT_MODEL_DIRECTORY = os.path.join(str(Path.home()), ".cache", "gpt4all").replace("\\", "\\\\")

DEFAULT_CONFIG = {
    "max_tokens": "4096",
    "chunk_size": "2056",
    "chunk_overlap": "256",
    "k": "4",
    "temperature": "0.5",
    "model_path": DEFAULT_MODEL_DIRECTORY,
    "n_batch": "8",
}

LOADER_MAPPING = {
    ".csv": {
        "loader": CSVLoader,
        "args": {}
    },
    ".doc": {
        "loader": UnstructuredWordDocumentLoader,
        "args": {}
    },
    ".docx": {
        "loader": UnstructuredWordDocumentLoader,
        "args": {}
    },
    ".epub": {
        "loader": UnstructuredEPubLoader,
        "args": {}
    },
    ".md": {
        "loader": UnstructuredMarkdownLoader,
        "args": {}
    }
    # ".pdf": {
    #     "loader": PDFMinerLoader,
    #     "args": {}
    # }
}

for ext in ALLOW_FILES:
    if ext not in LOADER_MAPPING:
        LOADER_MAPPING[ext] = {
        "loader": TextLoader,
        "args": {}
        }