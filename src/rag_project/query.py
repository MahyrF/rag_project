#interroge l'index avec le LLM

#var
from rag_project import app_config as cfg
from rag_project import ingest as ing

import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def get_chain():
    embeddings = HuggingFaceEmbeddings(model_name=cfg.EMBEDDING_MODEL)
    db = FAISS.load_local(cfg.INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": cfg.TOP_K})

    llm = OllamaLLM(model=cfg.LLM_MODEL, base_url=cfg.OLLAMA_URL)

    prompt = PromptTemplate.from_template(
                f"""
                {cfg.PROMPT_MODEL}

                Contexte: {{context}}
                Question: {{question}}
                Réponse:
                """
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


if __name__ == "__main__":

    if cfg.INDEX_DIR.exists():
        shutil.rmtree(cfg.INDEX_DIR)

    cfg.INDEX_DIR.mkdir()

    fichiers = ing.select_files()
    n_chunks = ing.run_ingest(fichiers)
    print(f"\nDocuments prêts. {n_chunks} extraits indexés.")

    chain, retriever = get_chain()

    question = str(input("Votre question: "))

    print("\n**********************Chatbot**************************\n")
    print(f"\nQuestion posée : {question}")
    print("\nGénération de la réponse du chatbot en cours...\n")
    print(f"\nRéponse du chatbot : {chain.invoke(question)}")