import os
import sys
from dotenv import load_dotenv

if os.environ.get("LOAD_ENV") == "1":
    load_dotenv(override=True)

BASE_PATH = os.environ.get('BASE_PATH', '')
GITHUB_API_KEY = os.environ.get('GITHUB_API_KEY', None)
PINECONE_INDEX = os.environ.get('PINECONE_INDEX', None)
PINECONE_KEY = os.environ.get('PINECONE_KEY', None)
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', None)
LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
EMBEDDINGS_MODEL = os.environ.get('EMBEDDINGS_MODEL', 'all-MiniLM-L6-v2')
LANGUAGE_MAP = {
    'py': 'python',
    'js': 'javascript',
    'java': 'java',
    'rb': 'ruby',
    'php': 'php',
    'cs': 'c#',
    'c': 'c',
    'cpp': 'c++',
    'go': 'go',
    'rs': 'rust',
    'scala': 'scala',
    'swift': 'swift',
    'kt': 'kotlin',
    'ts': 'typescript',
    'dart': 'dart',
    'r': 'r',
    'sh': 'bash',
    'html': 'html',
    'css': 'css',
    'json': 'json',
    'xml': 'xml',
    'yml': 'yaml',
    'yaml': 'yaml',
    'toml': 'toml',
    'ini': 'ini',
    'sql': 'sql',
    'tex': 'latex',
    'md': 'markdown',
    'txt': 'text'
}

if GITHUB_API_KEY is None:
    print('GITHUB_API_KEY not found in .env file')
    sys.exit(1)

if PINECONE_INDEX is None:
    print('PINECONE_INDEX not found in .env file')
    sys.exit(1)

if PINECONE_KEY is None:
    print('PINECONE_KEY not found in .env file')
    sys.exit(1)

if OPENAI_API_KEY is None:
    print('OPENAI_API_KEY not found in .env file')
    sys.exit(1)