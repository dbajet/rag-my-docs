# RAG My Docs

A demonstration of retrieval-augmented generation (RAG), step by step, on a fixed
set of IRS publications for tax year 2025. Ask a question; the app finds the
passages closest to it, sends them with the question to a language model, and
shows every step: the passages and their scores, the exact prompt, and the
answer citing the passages it used.

## What RAG does here

```
once, at startup:   download publications ─▶ extract text ─▶ sections ─▶ passages ─▶ embeddings

per question:       embed the question ─▶ nearest passages ─▶ prompt the model ─▶ answer with citations
```

1. **The corpus** (`CorpusCommand`). Twelve IRS publications (Pub. 17, 334, 463, 501,
   502, 503, 526, 527, 587, 590-A, 970, and the 1040 tax tables), public domain, are
   downloaded from irs.gov at startup. Each is checked against the SHA-256 pinned in
   `CorpusConstants`, so the test set's answers stay true; a revised publication is
   skipped until its checksum is updated.
2. **Extraction and sections** (`PdfBox`, `PageLayout`, `PageFurniture`,
   `SectionDetector`). PyMuPDF reads each page in reading order; running headers and
   page numbers are dropped; sections come from the PDF's bookmarks.
3. **Passages** (`Chunker`). A *passage* (chunk) is a few consecutive sentences,
   about 200 tokens, inside one section; table rows are kept whole with their header.
   *Inspect* shows every section and passage of a publication.
4. **Embeddings** (`EmbeddingBox`). An *embedding* is a list of 384 numbers placed so
   that texts with similar meaning are close. A small local model
   ([`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5)) embeds
   each passage under its publication and section names. Vectors live in PostgreSQL
   (pgvector).
5. **Retrieval** (`SearchStore`). The question is embedded by the same model, and the
   *k* passages closest to it by cosine similarity are taken (you choose *k*: 1, 3, 5 or 10;
   on the test set, 1 loses answers, 5 finds all it can, 10 only lengthens the prompt).
   If even the closest scores under 0.66, the question is not about the publications:
   it is answered "not covered" without calling a model.
6. **Generation** (`AnswerCommand`). The passages, numbered, and the question go to the
   language model with instructions to answer from them alone and name the passages
   it used; those become the citations.

**What it does well**: questions whose answer sits in one or two passages ("What is
the 2025 standard mileage rate?"). **What it does badly, on purpose**: the tax tables
(Pub. 1040). Rows of numbers do not resemble a question, so retrieval by meaning
rarely finds the right line. The test set includes such a question, expected to fail.

## The test set

Nineteen questions with what a right answer needs (`EvaluationConstants`): a passage
from a publication that holds it, and the facts the answer must contain. Two cannot
be answered from the publications and expect the answer to say so. *Ask › Test set*
runs them all with the chosen number of passages, so the effect of *k*, of the model
or of the prompt is measured rather than guessed.

## The app

- **One user**: the address in `BOOTSTRAP_ADMIN_EMAIL` signs in with an emailed link.
  Any other account is disabled at startup.
- **Language models** in *Settings › Language models*: Claude (cloud) or any
  OpenAI-compatible endpoint, such as a local vLLM. Each connection has a green or red
  light; one is chosen to answer.
- Stored text is sealed at rest (Fernet, `DATA_ENCRYPTION_KEY`), from when the app kept
  private documents.

## Running it locally

```bash
cp .env.example .env && chmod 600 .env       # APP_ENV=dev; leave SMTP_HOST empty
docker run -d --name rag_my_docs-dev-db -p 127.0.0.1:5540:5432 \
    -e POSTGRES_DB=db_rag_my_docs -e POSTGRES_USER=user_rag_my_docs -e POSTGRES_PASSWORD=dev \
    pgvector/pgvector:0.8.7-pg16
DATABASE_URL=postgresql://user_rag_my_docs:dev@127.0.0.1:5540/db_rag_my_docs \
BOOTSTRAP_ADMIN_EMAIL=you@example.com \
    uv run uvicorn rag_my_docs.main:app --app-dir app --port 8140
```

Open <http://localhost:8140> and ask for a sign-in link: without `SMTP_HOST` it is
written to the log. The first start downloads the embedding model (~130 MB) and the
publications (~21 MB), then embeds about 4,400 passages: a few minutes on a laptop
CPU. *Publications* shows the progress. Add a language model in *Settings* before
asking. Every setting is explained in `.env.example`.

## Checks

```bash
docker run -d --name rag_my_docs-test-db -p 127.0.0.1:55432:5432 \
    -e POSTGRES_DB=rag_my_docs_test -e POSTGRES_USER=rag_my_docs -e POSTGRES_PASSWORD=rag_my_docs \
    pgvector/pgvector:0.8.7-pg16
export TEST_DATABASE_URL=postgresql://rag_my_docs:rag_my_docs@127.0.0.1:55432/rag_my_docs_test

uv run ruff check .            # lint
uv run ruff format --check .   # formatting
uv run mypy .                  # strict types
uv run pytest                  # tests, 100% coverage required
```

CI runs the same four on every push; a deploy refuses a commit that did not pass.

## Code layout

```
app/rag_my_docs/
├── main.py        # AppFactory: builds and wires everything
├── handlers/      # HTTP routes and request models
├── commands/      # what the app does: corpus, answers, test set, sign-in…
├── libraries/     # database stores, the PDF pipeline, outside services (*Box)
├── structures/    # immutable data types
├── constants/     # settings that are not configuration, the corpus, the test set
└── static/        # the web app: plain HTML, CSS and JavaScript
tests/             # mirrors app/; tests/support/ holds fakes for outside services
deploy/            # deploy scripts, nginx site, backup container
```

## Deploying

One VM with Docker Compose, nginx and certbot. The app listens on the loopback only;
nginx is the only way in. Two copies run in turn (blue and green), so a release
starts beside the old one and takes over only once healthy.

1. Clone the repository on the server (`/opt/rag_my_docs`). Create `.env` from
   `.env.example` (`APP_ENV=production`, your domain, a fresh
   `DATA_ENCRYPTION_KEY`, the owner's address) and `.env.backup` from
   `.env.backup.example`, both `chmod 600`. **Keep a copy of the encryption key and
   the backup private key off the server**: a restored backup needs both.
2. Install `deploy/nginx-rag_my_docs.conf` as the site, with your domain, and run
   `certbot --nginx -d <domain>`. Give the deploy user that one file, and sudo
   rights for `nginx -t` and `nginx -s reload` only.
3. On your laptop, fill `deploy/local.env` (server, SSH key, a read-only GitHub
   token for CI status), then run `deploy/deploy-from-local.sh`.

**Rollback**: the deploy prints the commands: start the previous color again and
point nginx back to it. Database changes are additive, so the previous release still
runs.

**Backups**: a sidecar container dumps the database daily, encrypts the dump with
[age](https://age-encryption.org) to a key whose private half never touches the
server, and uploads it to S3. Set `BACKUP_HEARTBEAT_URL` to a monitor
(healthchecks.io…) that alerts when backups stop. The corpus can always be
downloaded again; what a backup keeps is the account and the model connections.

### Restore drill (every quarter)

Restores into a throwaway container, never over the live database:

```bash
(
set -euo pipefail
aws s3 cp s3://<bucket>/daily/rag_my_docs/<date>/db_rag_my_docs.dump.age .
age -d -i rag_my_docs-backup.key -o db_rag_my_docs.dump db_rag_my_docs.dump.age
docker run -d --name rag_my_docs-restore-drill -p 127.0.0.1:55499:5432 -e POSTGRES_PASSWORD=drill pgvector/pgvector:0.8.7-pg16
until docker exec rag_my_docs-restore-drill pg_isready -q -h 127.0.0.1 -U postgres; do sleep 1; done
docker exec -i rag_my_docs-restore-drill pg_restore -U postgres -d postgres --create --no-owner --exit-on-error < db_rag_my_docs.dump
docker exec rag_my_docs-restore-drill psql -U postgres -d db_rag_my_docs -c '\dt+'
)
```

Then start a local copy of the app against it
(`DATABASE_URL=postgresql://postgres:drill@127.0.0.1:55499/db_rag_my_docs` and the
production `DATA_ENCRYPTION_KEY`) and open a publication. Clean up with
`docker rm -f rag_my_docs-restore-drill && shred -u db_rag_my_docs.dump*`.

**Real recovery**: the same download and decryption; then stop both app colors,
`pg_restore --clean --if-exists` into the live database, and start the color nginx
points at. Up to a day of data can be lost.

## Limitations

- Retrieval by meaning misses rows of numbers (the tax tables) and questions that
  need many passages at once ("list every credit for…").
- A word hyphenated across two PDF lines stays split ("self-employ- ment").
- The embedding model is English-oriented.
- PyMuPDF is AGPL-3.0; this application's source is public.
