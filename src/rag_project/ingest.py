#charge les pdfs, crée l'index FAISS

#var
from rag_project import app_config as cfg

import sys
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def select_files():
    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()

    files = filedialog.askopenfilenames(
        title="Sélectionner les fichiers à interroger",
        filetypes=[
            ("PDF files", "*.pdf")
            ]
    )

    root.destroy()

    #dans le cas ou on a pas selectionné de fichiers
    if not files:
        print("\nAucun fichier sélectionné, arrêt du programme.")
        sys.exit()

    return list(files)


#NOTE: l'index (faiss_index/) n'est pas persisté entre les sessions par choix.
#Il est régénéré à chaque changement de corpus ou redémarrage de l'app pour éviter toute pollution entre différents jeux de documents.
#Ne pas "corriger" en ajoutant un volume Docker sans discussion préalable.


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
