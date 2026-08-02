FROM python:3.10-slim

#installer uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app


#Fichiers de config + README + métadonnées du package (changent rarement)
COPY pyproject.toml uv.lock ./
COPY src/rag_project/__init__.py src/rag_project/__init__.py
COPY README.md README.md

#Installer les dépendances (mis en cache tant que les fichiers ci-dessus ne changent pas)
RUN uv sync --frozen --no-dev --no-install-project

#copier le code après
COPY src/ src/


#build le projet lui-même
RUN uv sync --frozen --no-dev

CMD ["uv", "run", "streamlit", "run", "src/rag_project/app.py", "--server.address=0.0.0.0"]
