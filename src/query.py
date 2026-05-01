#interroge l'index avec le LLM

#var
from app_config import INDEX_DIR, EMBEDDING_MODEL, LLM_MODEL

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})

llm = OllamaLLM(model=LLM_MODEL)

prompt = PromptTemplate.from_template("""
Utilise uniquement le contexte suivant pour répondre à la question.
Si tu ne sais pas, dis-le.

Contexte: {context}
Question: {question}
Réponse:
""")

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

questions = [
    "Quel est le déficit public prévu pour 2026 ?",
    "Quels sont les principaux risques identifiés pour les finances publiques ?",
    "Quelle est la trajectoire de la dette publique ?",
]

for q in questions:
    print(f"\nQ: {q}")
    print(f"R: {chain.invoke(q)}")
