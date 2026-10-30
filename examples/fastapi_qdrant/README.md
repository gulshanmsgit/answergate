# A real RAG pipeline, and the pull request that broke it

A small but real RAG app: **FastAPI** serving answers about the **FastAPI documentation** (95 pages, MIT licence, pinned to release 0.162.0).

| Part | What it uses |
|---|---|
| Chunking | Paragraph-packed chunks of `CHUNK_SIZE` characters, each prefixed with its page title and section |
| Embeddings | `BAAI/bge-small-en-v1.5` via fastembed, on CPU |
| Vector store | Qdrant (local mode, or a server via `QDRANT_URL`) |
| Generation | `granite4.1:8b` via Ollama, temperature 0, told to answer only from the passages |
| Golden set | 44 hand-written questions: 40 answerable, each with a reference answer checked against the docs, and 4 it must decline |
| Judge | `granite4.1:8b`, which scored 32/32 on `ragcheck calibrate` |

## Run it

```bash
pip install -e ../.. -r requirements.txt
ollama pull granite4.1:8b
python fetch_docs.py                 # downloads the pinned docs into data/docs
uvicorn app:app --port 8000          # indexes into Qdrant on first start
ragcheck run -c ragcheck.yaml        # in another terminal
```

`bash experiment.sh 250 40` runs the whole comparison below: a baseline at the committed settings, then a run with `CHUNK_SIZE=250, CHUNK_OVERLAP=40`.

## The experiment: chunks from 1,000 to 250 characters

It's a one-line change that looks harmless in review. Smaller chunks are often recommended for "more precise retrieval". Measured on this pipeline, on one machine (Ollama on a consumer GPU), about 35 minutes per run:

| Metric | 1,000-char chunks | 250-char chunks | Change [95% CI] |
|---|---:|---:|---:|
| Hit rate | 0.975 | 0.975 | ±0 |
| Context precision | 0.625 | 0.725 | **+0.100** [+0.025, +0.175] |
| Faithfulness | 0.935 | 0.913 | −0.022 [−0.110, +0.069] |
| Correctness | 0.925 | 0.856 | **−0.069** [−0.131, −0.019] |
| Answer relevance | 0.988 | 0.900 | **−0.087** [−0.169, −0.025] |
| No PII leaked / refuses off-topic | 1.000 / 1.000 | 1.000 / 1.000 | ±0 |

**RAGCheck failed the change:**
- correctness and relevance both dropped more than the allowed 0.05;
- in both cases the whole 95% interval lies below zero, so this isn't noise;
- 19 of 44 questions got worse.

**Why.** Retrieval looked *better*: the right page was still found, and precision rose because more of the 4 chunks came from it. But a 250-character chunk often cuts the fact itself off. For example:
- the line saying `CORSMiddleware`'s `allow_methods` defaults to `['GET']`;
- the sentence naming `pydantic-settings`.

The model then honestly answers "I don't know based on the FastAPI docs."

A retrieval-only dashboard would have approved this PR. It takes answer-level checks against reference answers to see it.

## What the real pipeline taught RAGCheck

The first run on real documents found three bugs in RAGCheck itself, now fixed with regression tests:

| Bug | Seen in | Fix |
|---|---|---|
| IP addresses and `#ids` flagged as phone numbers | `--forwarded-allow-ips="192.168.1.100"`, `Client #1596980209979` | dotted IPv4 and `#`-prefixed numbers are not phones |
| "I don't know…, the passages discuss X" counted as answering an off-topic question | the brute-force question | judge prompt and heuristic both count an explained decline as declining |
| True claims rejected as "evidence not found" | `openssl rand -hex 32` | the evidence check ignores the passage labels the judge copies, and accepts a quote whose lines all appear in the context |

**Effect on the baseline:**
- faithfulness went from 0.881 to 0.935;
- the PII check from 0.977 to 1.000;
- refusals from 0.750 to 1.000.

The judge isn't perfect either. Once in 88 judged answers, granite garbled its own quote at the characters `["*`. The evidence check rejected that quote, as designed, so the case reads as unfaithful. That's the noise the confidence intervals exist for.

## The pull-request check

[`.github/workflows/rag-pr.yml`](../../.github/workflows/rag-pr.yml) runs on a pull request that changes this folder. It:
1. starts the app and runs RAGCheck against the committed baseline ([`.ragcheck/baseline.json`](.ragcheck/baseline.json));
2. posts or updates one PR comment headed by what broke and the worst questions;
3. uploads the HTML report;
4. fails the check on a regression.

It runs on a **self-hosted runner** with Ollama, so documents and answers stay on that machine. A self-hosted runner executes the PR's code, so the job only runs for branches of the same repository, never forks.
