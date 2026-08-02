#CONSTANTES
import os
from pathlib import Path

#Chemins d'accès (path)
BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR = BASE_DIR / "faiss_index"

#Modeles
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "mistral"
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

#prompt
PROMPT_MODEL = """Utilise uniquement le contexte suivant pour répondre à la question.
Si tu ne sais pas, dis-le clairement."""