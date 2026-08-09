# Weather RAG Pipeline

Same end-to-end pattern as the ticker-news pipeline (harvest → vectorize →
retrieve), applied to a new unstructured data source: weather.

## Data source

**National Weather Service API** (`api.weather.gov`) — free, no API key,
generous rate limits, and returns rich free-text narrative (alert
`description`/`instruction` fields, and forecast `detailedForecast` per
period) rather than structured numeric data, which is what this exercise
needed to embed.

**Geocoding**: locations are given as `"City, ST"` or `"lat,lon"`. City/state
inputs are resolved via **Open-Meteo's geocoding API**, not Nominatim/OSM —
Nominatim's usage policy blocks/deprioritizes traffic from cloud/datacenter
IP ranges (AWS, GCP, Azure), and Databricks Apps run on exactly that kind of
infrastructure, so Nominatim consistently 403'd server-side calls in testing.
Open-Meteo is free, keyless, and built for city-name lookups.

**Alerts by point, not by state**: the assignment brief suggests
`GET /alerts/active?area={state}`, but this pipeline uses
`GET /alerts/active?point={lat},{lon}` instead, since locations are specific
cities — point-based lookup returns only alerts that actually apply to that
location rather than the entire state.

## Schema

**`weather_documents`** (raw harvested text):
| column | type | notes |
|---|---|---|
| `id` | `TEXT PK` | NWS alert `id`, or a hash of `location+period+start_time` for forecasts (NWS doesn't give forecast periods a stable id) |
| `location` | `TEXT` | as passed to `/weather/sync` |
| `source_type` | `TEXT` | `"alert"` or `"forecast"` |
| `headline` | `TEXT` | event name / period name |
| `narrative_text` | `TEXT` | the text that gets embedded |
| `issued_at` | `TIMESTAMPTZ` | |
| `payload` | `JSONB` | raw API response, for provenance |
| `synced_at` | `TIMESTAMPTZ` | |

**`weather_embeddings`** (vectors):
| column | type | notes |
|---|---|---|
| `id` | `TEXT PK` | `"{document_id}_{chunk_index}"` |
| `document_id` | `TEXT` | FK → `weather_documents.id` |
| `chunk_index` | `INT` | |
| `chunk_text` | `TEXT` | |
| `embedding` | `vector(384)` | pgvector, HNSW index, cosine ops |
| `model_name` | `TEXT` | |
| `embedded_at` | `TIMESTAMPTZ` | |

**Chunking**: sliding window, `CHUNK_SIZE=800` / `CHUNK_OVERLAP=100` chars —
same values as the ticker-news pipeline. Most NWS alert/forecast text is
well under 800 chars, so most documents produce exactly one chunk; this only
kicks in for longer combined alert description+instruction text.

**Embedding model**: `sentence-transformers/all-MiniLM-L6-v2`, 384-dim — same
model as the ticker-news pipeline, so both stay queryable with the same
distance operator conventions.

## Running the pipeline end-to-end

1. **Harvest**: `POST /weather/sync` with `{"locations": ["Chicago, IL", "Austin, TX"]}` — fetches alerts + forecasts, upserts into `weather_documents`.
2. **Set up the vector table** (one-time): run `sql/setup_weather_embeddings_table.sql` in the Lakebase SQL Editor, with `{{EMBEDDING_DIM}}` replaced by `384`.
3. **Vectorize**: run `notebooks/ingest_weather_embeddings.py` — reads unembedded rows from `weather_documents`, chunks + embeds them, writes to `weather_embeddings`. Finish by running the printed `UPDATE ... embedding = embedding::vector` statement once in the SQL Editor.
4. **Retrieve**: `POST /weather/search` with `{"query": "flash flood risk this weekend", "top_k": 5}` — returns the top-k most similar chunks by cosine similarity.

## Known limitations / what I'd improve

- **Stale embeddings on forecast revision**: the ingestion script only embeds documents with *no* existing embedding row (`LEFT JOIN ... WHERE e.id IS NULL`). If a forecast's `narrative_text` is revised after it's already been embedded, the stale embedding isn't refreshed. A content-hash comparison instead of existence-check would fix this.
- **Manual two-step vector cast**: embeddings are inserted as a Postgres array literal cast to `double precision[]`, then a separate manual `UPDATE ... ::vector` statement is required — binding a Python list directly to a `vector` column via psycopg2 wasn't reliable in this environment. Would be cleaner as a single step if that's resolved.
- **No retry/backoff on NWS or Open-Meteo calls**: a transient failure on one location just gets reported in `/weather/sync`'s `errors` array rather than retried.
- **Geocoding is single-result**: takes Open-Meteo's first (or first state-matching) result with no disambiguation UI for ambiguous city names.
