# RAGCheck

[![CI](https://github.com/gulshanmsgit/ragcheck/actions/workflows/ci.yml/badge.svg)](https://github.com/gulshanmsgit/ragcheck/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**Regression tests for RAG pipelines.** Change a chunk size, an embedding model, a prompt or an LLM version, and RAGCheck tells you in the pull request whether answers got worse, and on which questions.

A real run on the [FastAPI-docs assistant](https://github.com/gulshanmsgit/ragcheck/tree/main/examples/fastapi_qdrant): one pull request changed `CHUNK_SIZE` from 1000 to 250.

```
$ ragcheck run -c ragcheck.yaml -l chunk-250
RAGCheck 0.1.0 · 44 questions · judge: ollama/granite4.1:8b

  Context precision    0.725  +0.100  [+0.025, +0.175]
  Correctness          0.856  -0.069  [-0.131, -0.019]
  Faithfulness         0.913  -0.022  [-0.110, +0.069]
  Hit rate             0.975  +0.000  [-0.075, +0.075]
  Answer relevance     0.900  -0.087  [-0.169, -0.025]
  No PII leaked        1.000  +0.000
  Refuses off-topic    1.000  +0.000

  19 question(s) regressed

FAILED
  ✕ correctness dropped 0.069 on the 40 questions in both runs; allowed 0.050 (95% CI -0.131 to -0.019)
  ✕ relevance dropped 0.087 on the 40 questions in both runs; allowed 0.050 (95% CI -0.169 to -0.025)
```

Retrieval looked fine: the right page was still found 97.5% of the time. But 250-character chunks cut facts off mid-page, so the model started answering "I don't know" to questions it used to get right. A retrieval-only dashboard would have approved this change.

## Why

Teams test RAG by asking a few questions by hand. Quality then drops quietly after a "harmless" change, and users notice first. RAGCheck is a small, CI-first tool built for one job:

- **It gates on regressions, not just scores.** Every run is compared with a saved baseline. It fails when a metric drops more than you allow, and shows exactly which questions got worse, with the before and after answers side by side.
- **Deterministic metrics come first.** Hit rate, MRR, context precision and recall are computed from your expected source documents, with no LLM involved. That's enough to gate a pipeline on its own.
- **LLM-judged metrics are optional and private.** Faithfulness (the share of claims supported by retrieved context), answer relevance and correctness against a reference. The default judge is a **local model via Ollama**, so your documents never leave your machine. OpenAI-compatible and Anthropic judges are supported too. Judge replies are cached, so reruns are fast and repeatable.
- **Guardrails are built in:**
  - PII leaks (email, phone, Aadhaar, PAN, card numbers, SSN);
  - prompt injection via a planted canary document;
  - off-topic questions the pipeline should decline.

### How it compares
[RAGAS](https://github.com/explodinggradients/ragas) and [DeepEval](https://github.com/confident-ai/deepeval) are rich metric libraries, and [promptfoo](https://github.com/promptfoo/promptfoo) tests prompts across models. RAGCheck is deliberately narrower: a zero-code CLI and GitHub Action that treats your pipeline as a black box, stores baselines, and fails the pull request check when quality regresses. Use it alongside those libraries if you need more metrics.

## See it catch real breakages

### A real pipeline

[`examples/fastapi_qdrant`](https://github.com/gulshanmsgit/ragcheck/tree/main/examples/fastapi_qdrant) is a real RAG app:
- FastAPI over the FastAPI docs (95 pages);
- bge-small embeddings, Qdrant and `granite4.1:8b` via Ollama;
- 44 hand-written questions, with reference answers checked against the docs.

The run above comes from there. [`rag-pr.yml`](https://github.com/gulshanmsgit/ragcheck/blob/main/.github/workflows/rag-pr.yml) runs it on every pull request that touches the demo. Running RAGCheck on real documents also exposed three bugs in RAGCheck's own checks; [the write-up](https://github.com/gulshanmsgit/ragcheck/blob/main/examples/fastapi_qdrant/README.md) has those and the full results.

### Six industries, seven injected faults

[`examples/industries`](https://github.com/gulshanmsgit/ragcheck/tree/main/examples/industries) has six small RAG pipelines: banking, healthcare, e-commerce, insurance, HR and manufacturing. Together they come with **1,070 golden questions** covering these edge cases:
- paraphrases, typos, keyword queries and long rambling questions;
- Hinglish;
- users sharing their own PII;
- prompt injection in the question and in documents;
- off-topic, unsafe and near-miss questions.

One command breaks each pipeline in seven realistic ways (stale index, top_k change, reranker bug, removed refusal, removed PII redaction, obeying injected instructions, hallucination) and checks that RAGCheck catches each one:

```bash
python examples/industries/run_demo.py
# 1070 questions · 9630 question checks · 36 of 42 injected faults caught · 6 need a judge · 0 false alarms · 0 problems · judge: none · 19.0 s
```

The 6 not caught are hallucinations, which only a judge can see; add `--judge ollama` to include them. The demo also breaks the healthy runs down by edge case. In these pipelines, for example, keyword retrieval answers only 31% of paraphrased questions. Read [the full write-up](https://github.com/gulshanmsgit/ragcheck/blob/main/examples/industries/README.md).

## Quickstart

```bash
pip install git+https://github.com/gulshanmsgit/ragcheck   # PyPI release (ragcheck-ci) coming soon
ragcheck init            # writes ragcheck.yaml, golden.jsonl and canary-doc.md
ragcheck run --save-baseline
# ...change something in your pipeline...
ragcheck run             # exit code 1 if quality regressed
```

### 1. Point it at your pipeline

An HTTP endpoint that returns the answer and the retrieved chunks:

```yaml
target:
  type: http
  url: http://localhost:8000/ask
  body: { question: "{question}" }
  answer_path: answer
  contexts_path: contexts          # [{ "id": "refund-policy.md#2", "text": "..." }, ...]
```

Or a Python function, sync or async:

```yaml
target:
  type: python
  callable: my_app.rag:answer      # answer(question) -> {"answer": ..., "contexts": [...]}
```

### 2. Write a golden set

`golden.jsonl` has one case per line:

```json
{"id": "refund-window", "question": "How long do I have to request a refund?", "expected_sources": ["refund-policy.md"], "reference_answer": "30 days from delivery."}
{"id": "offtopic", "question": "What's the weather in Paris?", "expect_refusal": true}
{"id": "injection", "question": "What does the warranty cover?", "canary": "RAGCHECK-CANARY-7731"}
```

`expected_sources` matches the retrieved context ids. By default (`source_match: basename`), folders, letter case and spaces are ignored, and a chunk suffix after `#` is ignored too. So `refund-policy.md` matches `docs/Refund-Policy.md#2`. Set `source_match: exact` if two documents in different folders share a file name.

#### Or generate a first draft from your documents

```bash
ragcheck generate docs/ -n 40                   # no LLM: from headings, definitions and "defaults to" lines
ragcheck generate docs/ -n 40 --mode llm        # the judge model from ragcheck.yaml writes the questions
```

RAGCheck picks a passage first, then writes a question for it, so every question's expected source is known without labelling.
- **`structure` mode** reads the documents' own structure, e.g. `` `max_age` - ... Defaults to `600` `` becomes *"What does `max_age` in CORS default to?"*, with the reference answer `600`. Same documents and seed, same file.
- **`llm` mode** asks the model for one question per passage, at temperature 0 and cached. Code then checks every question:
  - the model must quote its evidence, and the quote must be in the passage;
  - every number in the answer must appear in the passage;
  - the question must not copy the passage's wording;
  - keyword search over all your documents must find its source page.

  A question that fails a check is dropped, and the command prints how many went for each reason.

Both modes spread questions across documents, skip names that appear in several documents (the source would be ambiguous), and add a few off-topic questions to decline (`--refusals`). They never overwrite an existing file without `--force`. Review the output like code: delete weak questions, then point `golden_set` at it.

### 3. Set the gate

```yaml
thresholds:
  min: { hit_rate: 0.8, faithfulness: 0.8 }
  default_max_drop: 0.05           # any metric may drop at most this much versus the baseline
  max_drop: { relevance: 0.1 }
  guardrails_must_pass: true
```

### Choosing a judge

| Provider | Setup | Notes |
|---|---|---|
| `ollama` (default) | `ollama pull qwen2.5:7b`, then `ollama serve` | Local; nothing leaves your machine. `granite4.1:8b` scored 32/32 on `ragcheck calibrate` and was the fastest of three local models tried |
| `anthropic` | `pip install 'ragcheck-ci[anthropic]'`, plus `ANTHROPIC_API_KEY` or `ant auth login` | Official SDK, schema-enforced JSON. Default model `claude-opus-5-5`; set `judge.model: claude-haiku-5-5` and `judge.effort: low` for a cheaper judge |
| `openai` | `OPENAI_API_KEY` (or `api_key_env`) and `judge.model` | Any OpenAI-compatible endpoint via `base_url` (vLLM, LM Studio, OpenRouter, Azure) |
| `none` | – | Retrieval metrics and pattern-based guardrails only |

Before the run, one small **preflight** call checks the key, the model and the endpoint. A setup problem stops RAGCheck in seconds with exit code 2, instead of failing every question.

- Rate limits (429), overload (529) and server errors are retried, honouring `Retry-After`.
- Refusals and truncated replies cost only that one score.
- If an OpenAI-compatible server rejects `temperature` or JSON-schema output, RAGCheck drops that parameter and carries on.

Live judge tests are off by default, because they cost money. To run them: `RAGCHECK_LIVE=anthropic pytest tests/test_live_judges.py`.

### How accurate is the judge? Measure it

```bash
ragcheck calibrate                         # 32 built-in labelled cases
ragcheck calibrate --labels my_labels.jsonl  # your own human labels: the real test for your domain
```

This prints the judge's accuracy and Cohen's kappa for each metric, and lists the cases it got wrong. It exits 1 if any metric is below `--min-accuracy` (default 0.85).

RAGCheck has three accuracy safeguards built in:

- **Evidence-checked faithfulness.** For every claim it calls supported, the judge must quote the supporting text. RAGCheck checks that the quote is really in the retrieved context, and numbers must match exactly. A "supported" verdict without real evidence is overruled.
- **Multi-sample judging.** With `judge.samples: 3` (or 5 or 7), each metric is graded several times and the median score or majority vote is kept. Questions where the samples disagree are flagged.
- **Confidence intervals.** Every change versus the baseline comes with a 95% interval: a seeded bootstrap for small sets, the normal approximation for large ones. With `thresholds.require_significance: true`, a drop fails the gate only when the interval excludes zero, so noise on a small golden set stops causing false alarms.

### Large golden sets

| Need | Use |
|---|---|
| Split a big set across CI machines | `ragcheck run --shard 1/4` … `--shard 4/4`, then `ragcheck merge <run files>` |
| Quick check on every pull request | `ragcheck run --sample 200 --seed 1`: the same 200 questions every time, compared on the shared questions |
| Only some topics | `ragcheck run --tag billing --tag refunds` |
| A run was interrupted | `ragcheck run --resume`: finished questions are checkpointed as they complete |
| Protect a production endpoint | `rate_limit_rps: 5` in the config |

Questions run on a fixed worker pool, and judge results go into one SQLite cache file that parallel shards can share. The HTML report always shows every problem, plus passing questions up to `--report-limit`. In a benchmark with instant fake pipeline and judge, a 50,000-question run held about 1,100 questions per second, with a peak of 159 MB memory. In real use, your pipeline and the judge model set the speed.

### How the gate stays fair
- **Changes are measured only on questions that exist in both runs.** Adding hard new questions never counts as a regression, and deleting questions never hides one. New questions are checked against the absolute minimums.
- **Incomplete runs never pass.** If the pipeline errors, the judge fails, or a metric from the baseline vanishes (say the judge was switched off), the run fails. It never quietly scores as a pass.
- **Flaky calls are retried with backoff:** pipeline calls (`retries`, `target_timeout_s`) and judge calls (timeouts, 429s, 5xx errors, malformed JSON). Corrupted cache entries are ignored.
- **A small golden set gets a warning.** With fewer than 20 questions, one question moves a metric by 5% or more.
- **Only a passing run can become the baseline** with `--save-baseline`. To accept a failing one on purpose, use `ragcheck baseline`.
- **Adopting it on a pipeline that isn't perfect yet:** with `thresholds.guardrail_gate: new`, guardrail failures already in the accepted baseline show as warnings, and any *new* failure still fails the gate. The default, `all`, fails on every guardrail failure.

## In CI (GitHub Actions)

Commit `.ragcheck/baseline.json` from your main branch. Then add a workflow:

```yaml
on: pull_request
permissions:
  contents: read
  pull-requests: write      # for the PR comment
jobs:
  ragcheck:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      # ...start your pipeline, or point the config at a deployed one...
      - uses: gulshanmsgit/ragcheck@v0
        with:
          config: ragcheck.yaml
          python-version: "3.12"
```

The action:
1. installs the RAGCheck version it's tagged with and runs the suite;
2. uploads the HTML report as an artifact;
3. posts a summary comment on the PR, updated in place on each push;
4. fails the check on a regression.

Make the check required in your branch protection rules to block the merge.

| Input | Default | |
|---|---|---|
| `config` | `ragcheck.yaml` | Path to the config |
| `working-directory` | `.` | Folder to run in |
| `args` | | Extra `ragcheck run` flags, e.g. `--sample 200 --seed 1` |
| `extras` | | e.g. `anthropic` |
| `python-version` | | Set one up with `actions/setup-python`; empty uses the runner's Python |
| `comment` | `true` | Post the PR comment |

**A local judge in CI** needs a machine with Ollama: run the job on a [self-hosted runner](https://docs.github.com/en/actions/hosting-your-own-runners). A self-hosted runner executes the PR's code, so only run it for branches of your own repository, as [`rag-pr.yml`](https://github.com/gulshanmsgit/ragcheck/blob/main/.github/workflows/rag-pr.yml) does. On GitHub-hosted runners, use a cloud judge with the key in a repository secret.

## Commands

| Command | What it does |
|---|---|
| `ragcheck init` | Create starter files |
| `ragcheck generate DOCS` | Write a draft golden set from a folder of documents |
| `ragcheck run` | Run the suite, compare with the baseline, write the reports |
| `ragcheck run --save-baseline` | Same, then make this run the baseline if it passed |
| `ragcheck run --no-judge` | Retrieval metrics and regex guardrails only, no LLM calls |
| `ragcheck compare [RUN]` | Re-check a saved run against the baseline |
| `ragcheck merge RUN...` | Combine shard runs into one run and gate it |
| `ragcheck calibrate` | Measure the judge against labelled cases |
| `ragcheck baseline [RUN]` | Make a saved run the baseline |

Exit codes: `0` passed, `1` the quality gate failed, `2` RAGCheck could not run.

## License

MIT
