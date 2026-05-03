#CONSTANTES

import os

#Chemins d'accès (path)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")

#Modeles
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "mistral"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3