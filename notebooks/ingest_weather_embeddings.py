# Databricks notebook source
# MAGIC %md
# MAGIC # Ingest Weather Documents -> Vector Embeddings (Lakebase)
# MAGIC
# MAGIC This is the weather-practice counterpart to
# MAGIC `notebooks/ingest_ticker_news_embeddings.py`, following the same
# MAGIC conventions:
# MAGIC
# MAGIC 1. Reads `weather_documents` rows that don't have embeddings yet
# MAGIC    (populated by the Flask app's `POST /weather/sync` route - this
# MAGIC    notebook doesn't call the NWS API itself, it only embeds what's
# MAGIC    already been synced).
# MAGIC 2. Chunks each `narrative_text` (sliding window, `CHUNK_SIZE`/`CHUNK_OVERLAP`).
# MAGIC 3. Computes a sentence embedding for each chunk using
# MAGIC    `sentence-transformers/all-MiniLM-L6-v2` (384-dim).
# MAGIC 4. Writes them into `weather_embeddings` via psycopg2, using the same
# MAGIC    array-literal -> `::vector` cast workaround as the ticker-news
# MAGIC    notebook (binding a Python list directly to a `vector` column via
# MAGIC    psycopg2 is unreliable in this environment).
# MAGIC
# MAGIC It re-uses the SAME Lakebase secret (scope `database`, key `lakebase-url`)
# MAGIC that `lakebase.py` uses in the Flask app, so no extra secrets are needed.
# MAGIC
# MAGIC **Before running this notebook**, manually run
# MAGIC `sql/setup_weather_embeddings_table.sql` in the Lakebase SQL Editor
# MAGIC (with `{{EMBEDDING_DIM}}` replaced by your model's dimension).

# COMMAND ----------

# DBTITLE 1,Install required packages
# MAGIC %pip uninstall -y psycopg2 psycopg2-binary
# MAGIC %pip install -q sentence-transformers pandas

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config

# COMMAND ----------

dbutils.widgets.text("documents_table_name", "weather_documents", "Source table (raw weather documents)")
dbutils.widgets.text("embeddings_table_name", "weather_embeddings", "Destination table (vectors)")
dbutils.widgets.text("embedding_model", "sentence-transformers/all-MiniLM-L6-v2", "Embedding model")
dbutils.widgets.text("chunk_size", "800", "Narrative text chunk size (chars)")
dbutils.widgets.text("chunk_overlap", "100", "Narrative text chunk overlap (chars)")
dbutils.widgets.text("lakebase_secret_scope", "database", "Lakebase secret scope")
dbutils.widgets.text("lakebase_secret_key", "lakebase-url", "Lakebase secret key")

DOCUMENTS_TABLE_NAME = dbutils.widgets.get("documents_table_name")
EMBEDDINGS_TABLE_NAME = dbutils.widgets.get("embeddings_table_name")
EMBEDDING_MODEL_NAME = dbutils.widgets.get("embedding_model")
CHUNK_SIZE = int(dbutils.widgets.get("chunk_size"))
CHUNK_OVERLAP = int(dbutils.widgets.get("chunk_overlap"))
LAKEBASE_SECRET_SCOPE = dbutils.widgets.get("lakebase_secret_scope")
LAKEBASE_SECRET_KEY = dbutils.widgets.get("lakebase_secret_key")

# Same dimension switch as the ticker-news notebook, so swapping
# EMBEDDING_MODEL_NAME via the widget stays consistent with the table's
# vector(N) column - just keep this in sync with sql/setup_weather_embeddings_table.sql.
match EMBEDDING_MODEL_NAME:
    case "sentence-transformers/all-MiniLM-L6-v2":
        EMBEDDING_DIM = 384
    case "sentence-transformers/all-MiniLM-L12-v2":
        EMBEDDING_DIM = 384
    case "sentence-transformers/all-mpnet-base-v2":
        EMBEDDING_DIM = 768
    case "BAAI/bge-small-en-v1.5":
        EMBEDDING_DIM = 384
    case "BAAI/bge-base-en-v1.5":
        EMBEDDING_DIM = 768
    case _:
        raise ValueError(
            f"Unknown embedding model {EMBEDDING_MODEL_NAME!r} - add its output "
            "dimension to the match/case block above before running this notebook."
        )

print(f"Using model {EMBEDDING_MODEL_NAME!r} -> {EMBEDDING_DIM}-dim vectors")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolve the Lakebase connection info
# MAGIC
# MAGIC Same secret, same decoding scheme as `lakebase.py`: a single
# MAGIC base64-encoded Postgres URL stored in a Databricks secret scope. Parsed
# MAGIC into components (not passed as a raw DSN string) - psycopg2's DSN
# MAGIC parser is brittle if the stored secret was ever double-encoded, so
# MAGIC connecting via explicit host/port/dbname/user/password avoids that
# MAGIC entirely.

# COMMAND ----------

# DBTITLE 1,Parse Lakebase Connection Info
import base64
from urllib.parse import urlparse

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()


def get_lakebase_url() -> str:
    secret = w.secrets.get_secret(scope=LAKEBASE_SECRET_SCOPE, key=LAKEBASE_SECRET_KEY)
    return base64.b64decode(secret.value).decode("utf-8")


lakebase_url = get_lakebase_url()
parsed = urlparse(lakebase_url)

db_host = parsed.hostname
db_port = parsed.port or 5432
db_name = parsed.path.lstrip("/")
db_user = parsed.username
db_password = parsed.password

print("Connection details:")
print(f"  Host: {db_host}:{db_port}")
print(f"  Database: {db_name}")
print(f"  User: {db_user}")

# COMMAND ----------

# DBTITLE 1,Test psycopg2 connection
import psycopg2

print(f"Testing connection to {db_host}:{db_port}/{db_name}\n")

try:
    conn = psycopg2.connect(
        host=db_host, port=db_port, dbname=db_name,
        user=db_user, password=db_password,
        sslmode="require", connect_timeout=10,
    )
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {DOCUMENTS_TABLE_NAME}")
    count = cursor.fetchone()[0]
    print(f"✅ Connection successful! Found {count} rows in {DOCUMENTS_TABLE_NAME}")
    cursor.close()
    conn.close()
except Exception as e:
    import traceback
    print(f"❌ Connection failed: {e}")
    traceback.print_exc()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Database setup instructions
# MAGIC
# MAGIC Before continuing, manually run `sql/setup_weather_embeddings_table.sql`
# MAGIC in the Lakebase SQL Editor (replace `{{EMBEDDING_DIM}}` with the value
# MAGIC printed above - 384 for the default model).

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load weather documents that don't have embeddings yet

# COMMAND ----------

# DBTITLE 1,Load unembedded documents
import pandas as pd
import psycopg2

conn = psycopg2.connect(
    host=db_host, port=db_port, dbname=db_name,
    user=db_user, password=db_password, sslmode="require",
)

try:
    query = f"""
        SELECT d.id, d.location, d.source_type, d.headline, d.narrative_text
        FROM {DOCUMENTS_TABLE_NAME} d
        LEFT JOIN {EMBEDDINGS_TABLE_NAME} e ON e.document_id = d.id
        WHERE e.id IS NULL
          AND d.narrative_text IS NOT NULL
          AND TRIM(d.narrative_text) != ''
    """
    docs_df = pd.read_sql_query(query, conn)
    print(f"Loaded {len(docs_df)} weather documents without embeddings from {DOCUMENTS_TABLE_NAME}")
    display(docs_df.head(5))
finally:
    conn.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Chunk narrative text
# MAGIC
# MAGIC Most NWS alert/forecast text is well under `CHUNK_SIZE`, so a single
# MAGIC chunk per document is the common case; this only splits the longer
# MAGIC combined alert description+instruction text.

# COMMAND ----------

# DBTITLE 1,Chunk narrative_text (sliding window)
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - overlap
    return [c for c in chunks if c]


out_document_ids, out_chunk_indexes, out_chunk_texts = [], [], []

for _, row in docs_df.iterrows():
    for chunk_index, chunk in enumerate(chunk_text(row["narrative_text"])):
        out_document_ids.append(row["id"])
        out_chunk_indexes.append(chunk_index)
        out_chunk_texts.append(chunk)

chunks_df = pd.DataFrame({
    "document_id": out_document_ids,
    "chunk_index": out_chunk_indexes,
    "chunk_text": out_chunk_texts,
})

print(f"Split {len(docs_df)} documents into {len(chunks_df)} chunks")
display(chunks_df.head(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute chunk embeddings

# COMMAND ----------

# DBTITLE 1,Compute embeddings
import os

from sentence_transformers import SentenceTransformer

os.environ["HF_HOME"] = "/tmp/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/.cache/huggingface"
os.environ["HF_HUB_CACHE"] = "/tmp/.cache/huggingface"

print(f"Loading embedding model {EMBEDDING_MODEL_NAME}...")
model = SentenceTransformer(EMBEDDING_MODEL_NAME, cache_folder="/tmp/.cache/huggingface")

batch_size = 32
all_embeddings = []

for i in range(0, len(chunks_df), batch_size):
    batch = chunks_df.iloc[i:i + batch_size]
    vectors = model.encode(batch["chunk_text"].tolist(), show_progress_bar=False)
    all_embeddings.extend(vectors.tolist())
    if (i + batch_size) % 128 == 0:
        print(f"  Processed {min(i + batch_size, len(chunks_df))}/{len(chunks_df)} chunks")

embeddings_df = chunks_df.copy()
embeddings_df["embedding"] = all_embeddings

print(f"Computed {len(embeddings_df)} embeddings using {EMBEDDING_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Upsert embeddings into Lakebase
# MAGIC
# MAGIC Written as a Postgres array literal cast via `::double precision[]`,
# MAGIC same workaround as the ticker-news notebook, followed by the manual
# MAGIC cast to `vector` - binding a Python list straight to a `vector` column
# MAGIC via psycopg2 doesn't reliably work in this environment.

# COMMAND ----------

# DBTITLE 1,Insert embeddings using psycopg2
from datetime import datetime

from psycopg2.extras import execute_values

embeddings_df["id"] = embeddings_df["document_id"] + "_" + embeddings_df["chunk_index"].astype(str)
embeddings_df["model_name"] = EMBEDDING_MODEL_NAME
embeddings_df["embedded_at"] = datetime.now()

rows = embeddings_df.to_dict("records")

if rows:
    print(f"Inserting {len(rows)} embeddings into {EMBEDDINGS_TABLE_NAME}...")

    conn = psycopg2.connect(
        host=db_host, port=db_port, dbname=db_name,
        user=db_user, password=db_password, sslmode="require",
    )
    try:
        cursor = conn.cursor()

        insert_data = [
            (
                row["id"],
                row["document_id"],
                int(row["chunk_index"]),
                row["chunk_text"],
                "{" + ",".join(str(float(x)) for x in row["embedding"]) + "}",
                row["model_name"],
                row["embedded_at"],
            )
            for row in rows
        ]

        insert_sql = f"""
            INSERT INTO {EMBEDDINGS_TABLE_NAME} (
                id, document_id, chunk_index, chunk_text, embedding, model_name, embedded_at
            ) VALUES %s
            ON CONFLICT (id) DO NOTHING
        """
        template = "(%s, %s, %s, %s, %s::double precision[], %s, %s)"
        execute_values(cursor, insert_sql, insert_data, template=template, page_size=100)

        conn.commit()
        print(f"✅ Successfully inserted rows into {EMBEDDINGS_TABLE_NAME}")
        print("\nIMPORTANT: run this once in the Lakebase SQL Editor to cast the arrays to vectors:")
        print(f"  UPDATE {EMBEDDINGS_TABLE_NAME} SET embedding = embedding::vector WHERE embedding IS NOT NULL;")
    finally:
        cursor.close()
        conn.close()
else:
    print("No new chunks to embed - nothing to insert.")