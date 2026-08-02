# RAG Pipeline — Interrogation de documents PDF

## Description

Ce projet implémente un pipeline de type **RAG (Retrieval-Augmented Generation)** permettant d'interroger un ensemble de documents PDF à l'aide d'un modèle de langage local.

Le système fonctionne en deux étapes :
1. Une phase d'indexation des documents
2. Une phase d'interrogation via une interface ou en ligne de commande

L'objectif est de permettre à un modèle de répondre à des questions en s'appuyant uniquement sur le contenu des documents fournis, sans hallucination externe.

---

## Stack technique

- **LangChain** : orchestration du pipeline
- **FAISS** : stockage et recherche vectorielle
- **HuggingFace Embeddings** : génération des embeddings (`paraphrase-multilingual-MiniLM-L12-v2`)
- **Ollama + Mistral** : exécution locale du modèle de langage (gratuit, sans API externe)
- **PyMuPDF** : extraction du texte depuis les PDF
- **Streamlit** : interface utilisateur

---

PDF → Extraction de texte → Découpage en chunks → Embeddings → Index FAISS

### Phase 2 : Interrogation

Question → Embedding → Recherche sémantique → Contexte → LLM → Réponse

---


## Structure du projet


├── faiss_index/           # Index vectoriel généré (non versionné)
<br>├── src/
<br>│   ├── app_config.py      # Configuration centralisée
<br>│   ├── ingest.py          # Pipeline d'indexation
<br>│   ├── query.py           # Pipeline d'interrogation
<br>│   └── app.py             # Interface Streamlit
<br>├── requirements.txt
<br>└── README.md

---

## Prérequis

- Python 3.10+
- [UV](https://docs.astral.sh/uv/) installé sur la machine:
  
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

- [Ollama](https://ollama.com) installé sur la machine:
  
```bash
sudo snap install ollama
```

## Installation

### Option 1 : Avec UV
```bash
git clone https://github.com/MahyrF/rag_project.git
cd rag_project
uv sync

ollama pull mistral
```
#### Lancer l'application

```bash
uv run streamlit run src/rag_project/app.py
```


---

### Option 2 : Avec Docker

#### Prérequis

- Ollama doit écouter sur toutes les interfaces réseau, pas uniquement `localhost`, 
  pour être accessible depuis le conteneur :
```bash
  sudo snap set ollama host=0.0.0.0:11434
  sudo snap set ollama origins='*'
  sudo systemctl restart snap.ollama.listener.service
```

#### Lancer l'application

```bash
docker compose up
```




## Utilisation

### Interface Streamlit (recommandé)


1. Chargez vos fichiers PDF via le panneau latéral
2. Cliquez sur **Analyser les documents**
3. Posez vos questions dans le champ de saisie

### En ligne de commande

```bash
uv run python -m src.rag_project.query
```

---

## Fonctionnement détaillé

### Découpage en chunks

Les documents sont découpés en segments de taille fixe avec chevauchement pour conserver le contexte entre deux morceaux consécutifs.

### Embeddings

Chaque chunk est transformé en vecteur numérique qui capture son sens sémantique. Deux textes similaires produiront des vecteurs proches, ce qui permet une recherche par similarité plutôt que par mot-clé exact.

### Recherche vectorielle

À chaque question, le retriever récupère les k chunks les plus proches sémantiquement et les transmet au LLM comme contexte.

### Génération

Le modèle reçoit le contexte extrait et la question, et génère une réponse ancrée sur les documents. Le prompt est conçu pour limiter les réponses aux informations présentes dans les documents.

---

## Configuration

Dans `src/app_config.py` :

| Paramètre | Description |
|---|---|
| `INDEX_DIR` | Dossier de l'index FAISS |
| `EMBEDDING_MODEL` | Modèle d'embedding HuggingFace |
| `LLM_MODEL` | Modèle Ollama utilisé |
| `PROMPT_MODEL` | Instructions pour le comportement du modèle |
| `CHUNK_SIZE` | Taille des chunks en tokens |
| `CHUNK_OVERLAP` | Chevauchement entre chunks |
| `TOP_K` | Nombre de chunks récupérés par requête |

---

## Améliorations possibles

- Support multilingue avec un modèle d'embedding adapté
- Déploiement via FastAPI

### Note sur la sécurité (dev local)

Cette configuration expose Ollama sur toutes les interfaces réseau (`0.0.0.0`) 
pour permettre la communication avec le conteneur Docker. C'est adapté à un usage 
en développement local, mais pas recommandé tel quel en production : idéalement, 
Ollama devrait tourner dans son propre conteneur sur un réseau Docker interne, 
sans port exposé à l'extérieur.
