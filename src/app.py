import os
import tempfile
import streamlit as st
from langchain_ollama import OllamaLLM


import app_config as cfg
import ingest as ing
import query as qy

st.set_page_config(page_title="RAG Finance Publique", page_icon="📄", layout="wide")
st.title("Interrogation de documents")

# ── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader(
        "Sélectionner un ou plusieurs fichiers PDF",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} fichier(s) chargé(s)")

        if st.button("Analyser les documents", use_container_width=True):
            with st.spinner("Lecture et indexation en cours, merci de patienter..."):
                tmp_paths = []
                tmp_dir = tempfile.mkdtemp()

                for f in uploaded_files:
                    tmp_path = os.path.join(tmp_dir, f.name)
                    with open(tmp_path, "wb") as out:
                        out.write(f.read())
                    tmp_paths.append(tmp_path)

                n_chunks = ing.run_ingest(tmp_paths)
                st.session_state["indexed"] = True
                st.success(f"Documents prêts. {n_chunks} extraits indexés.")

    st.divider()
    st.caption("Modèle : " + cfg.LLM_MODEL)
    st.caption("Embeddings : " + cfg.EMBEDDING_MODEL)

# ── MAIN ───────────────────────────────────────────────────
if not st.session_state.get("chain"):
    st.info("Chargez et indexez des documents pour commencer, ou posez une question générale.")

index_existe = os.path.exists(os.path.join(cfg.INDEX_DIR, "index.faiss"))

if "chain" not in st.session_state and index_existe:
    with st.spinner("Chargement du modèle..."):
        chain, retriever = qy.get_chain()
        st.session_state["chain"] = chain
        st.session_state["retriever"] = retriever

if st.session_state.get("indexed"):
    with st.spinner("Rechargement du modèle..."):
        chain, retriever = qy.get_chain()
        st.session_state["chain"] = chain
        st.session_state["retriever"] = retriever
        st.session_state["indexed"] = False

# ── HISTORIQUE ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── INPUT ──────────────────────────────────────────────────
question = st.chat_input("Posez votre question sur les documents...")

if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        if "chain" in st.session_state:
            with st.spinner("Recherche en cours..."):
                response = st.session_state["chain"].invoke(question)
                sources = st.session_state["retriever"].invoke(question)
            st.write(response)
            with st.expander("Voir les extraits utilisés pour cette réponse"):
                for doc in sources:
                    nom_fichier = os.path.basename(doc.metadata.get('source', 'inconnu'))
                    st.markdown(f"**{nom_fichier}** — page {doc.metadata.get('page', '?')}")
                    st.caption(doc.page_content[:300] + "...")
        else:
            with st.spinner("Réflexion en cours..."):
                llm = OllamaLLM(model=cfg.LLM_MODEL)
                response = llm.invoke(question)
            st.write(response)

    st.session_state["messages"].append({"role": "assistant", "content": response})