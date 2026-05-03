#charge les pdfs, crée l'index FAISS

#var
import app_config as cfg

import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def run_ingest(pdf_paths: list[str]) -> int:
    docs = []
    for path in pdf_paths:
        print(f"Chargement : {path}")
        loader = PyMuPDFLoader(path)
        docs.extend(loader.load())
    print(f"{len(docs)} pages chargées")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.CHUNK_SIZE,
        chunk_overlap=cfg.CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"{len(chunks)} chunks créés")

    embeddings = HuggingFaceEmbeddings(model_name=cfg.EMBEDDING_MODEL)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(cfg.INDEX_DIR)
    print("Index sauvegardé.")

    return len(chunks)


if __name__ == "__main__":
    pdf_paths = [
        os.path.join(cfg.DOCS_DIR, f)
        for f in os.listdir(cfg.DOCS_DIR)
        if f.endswith(".pdf")
    ]
    run_ingest(pdf_paths)
