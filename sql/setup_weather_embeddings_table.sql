-- sql/setup_weather_embeddings_table.sql
--
-- Run this manually (Databricks Lakebase SQL Editor, connected to your
-- Lakebase instance) BEFORE running notebooks/ingest_weather_embeddings.py.
--
-- Replace {{EMBEDDING_DIM}} below with your model's output dimension.
-- sentence-transformers/all-MiniLM-L6-v2 (the default) -> 384

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS weather_embeddings (
    id TEXT PRIMARY KEY,                    -- "{document_id}_{chunk_index}"
    document_id TEXT NOT NULL REFERENCES weather_documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector({{EMBEDDING_DIM}}),
    model_name TEXT NOT NULL,
    embedded_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_weather_embeddings_document_id
    ON weather_embeddings (document_id);

-- HNSW index for cosine-similarity retrieval (Part 3's /weather/search endpoint)
CREATE INDEX IF NOT EXISTS idx_weather_embeddings_hnsw
    ON weather_embeddings
    USING hnsw (embedding vector_cosine_ops);