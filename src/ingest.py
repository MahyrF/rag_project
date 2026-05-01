#charge les pdfs, crée l'index FAISS

#var
import app_config as cfg

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os


docs = []
for f in os.listdir(cfg.DOCS_DIR):
    if f.endswith(".pdf"):
        print(f"Chargement : {f}")
        loader = PyMuPDFLoader(f"docs/{f}")
        docs.extend(loader.load())

print(f"{len(docs)} pages chargées")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"{len(chunks)} chunks créés")

print("Calcul des embeddings ...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = FAISS.from_documents(chunks, embeddings)
db.save_local(cfg.INDEX_DIR)
print("Index sauvegardé.")
