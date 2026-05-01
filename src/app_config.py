#CONSTANTES

import os

#Chemins d'accès (path)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
INDEX_DIR = os.path.join(BASE_DIR, "faiss_index")

#Modeles
LLM_MODEL = "mistral"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
