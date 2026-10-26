# Industry demo: break six RAG pipelines and watch RAGCheck catch it

Six small but real RAG pipelines (BM25 retrieval over Markdown documents, then an extractive answer), one per industry. Each has its own documents, a 22–24 question golden set and a config. Every company, person, email and phone number is invented.

| Industry | Company (fictional) | Questions about |
|---|---|---|
| `banking` | Northwind Bank | savings, fixed deposits, loans, cards and fraud, complaints |
| `healthcare` | CityCare Hospital | appointments, visiting hours, billing, records (admin only, no medical advice) |
| `ecommerce` | ShopKart | returns, delivery, payments, membership, support |
| `insurance` | SecureLife | motor and health cover, claims, renewals, grievances |
| `hr` | Acme Technologies | leave, hybrid work, payroll, learning, conduct and exit |
| `manufacturing` | GreenForge Components | product carbon footprints (ISO 14067), quality, suppliers, lead times, EHS |

Each golden set includes:
- single-document and multi-document questions;
- 2 off-topic questions the pipeline must decline;
- 1 question whose document hides a prompt injection;
- 1 question whose document holds a staff email and phone number, which must not leak.

## Run it

```bash
pip install -e .                                   # from the repo root
python examples/industries/run_demo.py             # all six industries, no LLM, a few seconds
```

For each industry the script:
1. Saves the healthy pipeline as the baseline. It must pass.
2. Runs the healthy pipeline again. It must still pass: no false alarm.
3. Switches in each fault below and runs again. It must fail, for the right reason.

It prints a scorecard and writes an HTML report per run to `examples/industries/out/<industry>/`. Measured without a judge on this machine:

```
36 of 42 injected faults caught · 6 need a judge · 0 false alarms · 0 problems · judge: none · 3.5 s
```

## The faults

Set `DEMO_FAULT` to switch one in. Each copies a real way RAG pipelines break:

| Fault | What changed | What trips |
|---|---|---|
| `stale_index` | An ingest job failed, so one document is missing from the index | hit rate, context precision and recall |
| `top_k_1` | Someone changed `top_k` from 3 to 1 | context recall and nDCG on multi-document questions |
| `rerank_bug` | The reranker sorts in the wrong direction | MRR and nDCG |
| `no_refusal` | The "I don't know" fallback was removed | the refusal guardrail |
| `pii_leak` | The redaction step was removed | the PII guardrail |
| `injectable` | The generator obeys instructions hidden in documents | the injection guardrail |
| `hallucinate` | Answers change numbers and add an unsupported claim | faithfulness and correctness: **needs a judge** |

## Try one by hand, like in CI

```bash
cd examples/industries
ragcheck run -c banking.yaml --save-baseline                  # healthy: PASSED, exit 0
DEMO_FAULT=pii_leak ragcheck run -c banking.yaml              # FAILED, exit 1
```

On Windows PowerShell, use `$env:DEMO_FAULT="pii_leak"` and `Remove-Item Env:DEMO_FAULT` afterwards.

## With a judge: catch hallucinations

Without a judge, `hallucinate` passes the gate. Retrieval is unchanged and the wrong numbers are not PII, so the scorecard reports it as `NEEDS JUDGE`, not as a catch. A judge reads the answer against the retrieved text:

```bash
python examples/industries/run_demo.py --judge ollama                        # needs `ollama serve` and `ollama pull qwen2.5:7b`
python examples/industries/run_demo.py --judge anthropic --model claude-haiku-5-5  # needs ANTHROPIC_API_KEY; costs money
python examples/industries/run_demo.py --judge ollama --industry banking --fault hallucinate   # one cell only
```

Every number in this file comes from a run without a judge. The judged results depend on the model you pick, so run it and read your own scorecard.

## What building this taught us (and changed in RAGCheck)

- **`context_recall` is now a default metric.** Without it, `top_k_1` barely moved the scores: losing the second document on multi-document questions only nudged nDCG.
- **Document-level labels can't see some retrieval bugs.** When all three retrieved sections come from the same document, scrambling their order changes nothing at document level. So the banking golden set labels the exact section (`savings-account.md#1`), and its `rerank_bug` is caught. The other industries use plain document names, which is what most teams start with.
- **The golden set decides what can be caught.** With only 2 multi-document questions out of 20, `top_k_1` moved recall by exactly the 0.05 allowed, and the gate passed. Add questions that need more than one document if retrieval depth matters to you.
- **The healthy pipelines aren't perfect, on purpose.** For example, the insurance pipeline declines "What must I do if my car is stolen?". A judge's relevance and correctness scores show this. The gate compares with the baseline, so a known weakness doesn't fail every run, but fixing it shows up as an improvement.
