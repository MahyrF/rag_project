# RAG Pipeline avec LangChain, FAISS et LLM local

## Description

Ce projet implémente un pipeline de type **RAG (Retrieval-Augmented Generation)** permettant d’interroger un ensemble de documents PDF à l’aide d’un modèle de langage.

Le système fonctionne en deux étapes :
1. Une phase d’indexation des documents
2. Une phase d’interrogation (query)

L’objectif est de permettre à un modèle de répondre à des questions en s’appuyant uniquement sur le contenu des documents fournis.

---

## Architecture

### Phase 1 : Indexation

PDF → Extraction de texte → Découpage → Embeddings → Index vectoriel

### Phase 2 : Interrogation

Question → Embedding → Recherche → Contexte → LLM → Réponse

---

## Stack technique

- LangChain : orchestration du pipeline
- FAISS : stockage et recherche vectorielle
- HuggingFace Embeddings : génération des embeddings
- Ollama : exécution locale du modèle de langage
- PyMuPDF : extraction du texte depuis les PDF

---

## Fonctionnement détaillé

### Chargement des documents

Les fichiers PDF sont parcourus depuis un dossier local et convertis en texte. Chaque page devient un document exploitable.

### Découpage en chunks

Les documents sont découpés en segments de taille fixe (par exemple 500 caractères) avec un chevauchement.

Objectifs :
- respecter les limites des modèles
- conserver le contexte
- améliorer la recherche

### Génération des embeddings

Chaque chunk est transformé en vecteur numérique.

Un embedding représente le sens d’un texte. Deux textes similaires auront des vecteurs proches.

### Indexation

Les vecteurs sont stockés dans une base FAISS qui est optimisée pour la recherche de similarité.

### Sauvegarde

L’index est sauvegardé localement pour éviter de recalculer les embeddings.

---

## Phase de requête

### Chargement de l’index

L’index vectoriel est rechargé avec le même modèle d’embedding utilisé lors de l’indexation.  
Cela garantit que les représentations vectorielles restent cohérentes entre les documents et les requêtes.

### Transformation de la question

La question utilisateur est injectée dans le pipeline.  
Elle est transmise telle quelle aux différentes étapes de traitement, ce qui permet un flux de données simple et explicite.

### Recherche

Le retriever interroge la base vectorielle pour récupérer les chunks les plus pertinents (top-k).  
Cette recherche repose sur la similarité sémantique entre la question et les documents.

### Formatage du contexte

Les documents récupérés sont transformés en un bloc de texte unique.  
Cette étape est explicitement définie dans le pipeline, ce qui permet de contrôler précisément la manière dont le contexte est construit.

### Construction du prompt

Le contexte et la question sont injectés dans un template.  
Le prompt est conçu pour forcer le modèle à répondre uniquement à partir des informations fournies.

### Génération

Le modèle de langage reçoit le prompt complet et génère une réponse.  
Chaque transformation (retrieval, formatage, injection) étant définie explicitement, le flux de données reste transparent et contrôlé.

### Résultat

Le système retourne une réponse générée à partir des documents les plus pertinents.

---

### Remarque

Cette implémentation repose sur une approche modulaire où chaque étape du pipeline est composée explicitement.  
Cela permet une meilleure flexibilité, une compréhension plus claire du traitement, et un contrôle fin sur chaque transformation.

---

## Structure du projet

.

├── docs/ # Documents PDF

├── faiss_index/ # Index FAISS

├── src/ # code source

|   ├── config.py # Configuration

|   ├── ingest.py # Script d’indexation

|   └── query.py # Script de requête

└── README.md


---

## Configuration

Dans `config.py` :

- `DOCS_DIR` : dossier des PDF
- `INDEX_DIR` : dossier de l’index
- `EMBEDDING_MODEL` : modèle d’embedding
- `LLM_MODEL` : modèle LLM utilisé

---

## Utilisation

### Indexation


python ingest.py


### Query

python query.py