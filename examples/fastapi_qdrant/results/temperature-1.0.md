<!-- answergate-report -->
### ❌ **AnswerGate failed**

**Answer consistency dropped 40% (0.848 → 0.509) · 27 of 44 questions got worse**

**What changed in the pipeline:** `temperature` 0 → 1

44 questions · p50 4476 ms · p95 5642 ms

**Why it failed**
- consistency dropped 0.339 on the 40 questions in both runs; allowed 0.050 (95% CI -0.407 to -0.270)

**Questions that got worse** (10 of 27)

| Question | What got worse | Answer now |
|---|---|---|
| How do I load a machine learning model once when the app starts and release it at shutdow… | Answer consistency 0.70 → 0.24<br>Answer relevance 1.00 → 0.00<br>Correctness 1.00 → 0.00 |  |
| How do I replace a dependency with a fake one during tests? | Answer consistency 0.74 → 0.45<br>Answer relevance 1.00 → 0.50<br>Correctness 1.00 → 0.25 | In a test you set the FastAPI app’s `dependency_overrides` dict to map the real dependency function to a fake one: ```python app.dependency… |
| How can I run some code after the response has been sent, like sending an email notificat… | Answer consistency 0.83 → 0.31<br>Faithfulness 0.67 → 0.22 | Use FastAPI **BackgroundTasks**. Add the task in your endpoint: ```python from fastapi import FastAPI, BackgroundTasks app = FastAPI() def… |
| How often does FastAPI send a keep-alive ping on a server-sent events stream? | Answer consistency 1.00 → 0.63<br>Faithfulness 1.00 → 0.50 | FastAPI inserts a keep‑alive ping comment automatically when no data has been sent for **15 seconds**. This “ping” comment prevents many pr… |
| How long do browsers cache CORS responses by default? | Answer consistency 1.00 → 0.48<br>Faithfulness 1.00 → 0.67 | Browsers cache the CORS pre‑flight response for **600 seconds** by default (the `max_age` default in the `CORSMiddleware`). This means the… |
| Which status code should an endpoint return after creating a new record? | Answer consistency 1.00 → 0.21 | An endpoint that creates a new record should return **201 “Created”**. The standard example shows a path operation that returns `status_cod… |
| Should test functions that use TestClient be async def? | Answer consistency 0.97 → 0.51<br>Answer relevance 1.00 → 0.75 | No. TestClient works only with normal `def` test functions; the “magic” that calls the async FastAPI app breaks inside an `async def`. If y… |
| What is special about the 204 status code? | Answer consistency 0.97 → 0.51<br>Faithfulness 1.00 → 0.75 | The **204** code means “No Content”. It’s used when the server has successfully processed the request but there is nothing to return, so th… |
| Which library and algorithm do the docs recommend for hashing passwords? | Answer consistency 1.00 → 0.32 | The docs recommend using the **`pwdlib`** library with the **Argon2** hashing algorithm. You can install it with: ```bash uv add "pwdlib[ar… |
| How do I serve an existing Flask app inside a FastAPI app? | Answer consistency 1.00 → 0.33 |  |

| Metric | Baseline | This run | Change [95% CI] |
|---|---:|---:|---:|
| Answer consistency | 0.848 | 0.509 | −0.339 [−0.407, −0.270] |
| Context precision | 0.625 | 0.625 | ±0 [±0, ±0] |
| Context recall | 0.975 | 0.975 | ±0 [±0, ±0] |
| Correctness | 0.887 | 0.869 | −0.019 [−0.100, +0.056] |
| Faithfulness | 0.890 | 0.917 | +0.027 [−0.051, +0.104] |
| Hit rate | 0.975 | 0.975 | ±0 [±0, ±0] |
| MRR | 0.892 | 0.892 | ±0 [±0, ±0] |
| nDCG | 0.913 | 0.913 | ±0 [±0, ±0] |
| Answer relevance | 0.925 | 0.894 | −0.031 [−0.100, +0.031] |
| No PII leaked | 1.000 | 1.000 | ±0 |
| Refuses off-topic | 1.000 | 1.000 | ±0 |

<details><summary>Every change (38)</summary>

| Case | Metric | Before | After |
|---|---|---:|---:|
| `cors-default-methods` | Answer consistency | 1.00 | 0.65 |
| `cors-max-age` | Answer consistency | 1.00 | 0.48 |
| `cors-max-age` | Faithfulness | 1.00 | 0.67 |
| `cors-credentials-wildcard` | Answer consistency | 1.00 | 0.53 |
| `gzip-level` | Answer consistency | 0.91 | 0.53 |
| `middleware-timer` | Answer consistency | 0.89 | 0.51 |
| `background-tasks` | Answer consistency | 0.83 | 0.31 |
| `background-tasks` | Faithfulness | 0.67 | 0.22 |
| `background-heavy` | Answer consistency | 0.84 | 0.50 |
| `testing-install` | Answer consistency | 1.00 | 0.39 |
| `testing-def` | Answer consistency | 0.97 | 0.51 |
| `testing-def` | Answer relevance | 1.00 | 0.75 |
| `dependency-overrides` | Answer consistency | 0.74 | 0.45 |
| `dependency-overrides` | Answer relevance | 1.00 | 0.50 |
| `dependency-overrides` | Correctness | 1.00 | 0.25 |
| `forms-install` | Answer consistency | 1.00 | 0.51 |
| `forms-and-json` | Answer consistency | 0.80 | 0.53 |
| `status-201` | Answer consistency | 1.00 | 0.21 |
| `status-204` | Answer consistency | 0.97 | 0.51 |
| `status-204` | Faithfulness | 1.00 | 0.75 |
| `exclude-unset` | Answer consistency | 1.00 | 0.47 |
| `path-ge` | Answer consistency | 0.81 | 0.49 |
| `path-ge` | Faithfulness | 1.00 | 0.75 |
| `path-enum` | Answer consistency | 0.88 | 0.47 |
| `path-enum` | Answer relevance | 1.00 | 0.75 |
| `docs-urls` | Answer consistency | 1.00 | 0.47 |
| `yield-cleanup` | Answer consistency | 1.00 | 0.54 |
| `password-hashing` | Answer consistency | 1.00 | 0.32 |
| `jwt-secret` | Answer consistency | 0.96 | 0.50 |
| `settings-cache` | Answer consistency | 0.98 | 0.56 |
| `settings-package` | Answer consistency | 0.90 | 0.42 |
| `lifespan` | Answer consistency | 0.70 | 0.24 |
| `lifespan` | Answer relevance | 1.00 | 0.00 |
| `lifespan` | Correctness | 1.00 | 0.00 |
| `websocket-disconnect` | Answer consistency | 1.00 | 0.41 |
| `wsgi-flask` | Answer consistency | 1.00 | 0.33 |
| `sse-keepalive` | Answer consistency | 1.00 | 0.63 |
| `sse-keepalive` | Faithfulness | 1.00 | 0.50 |

</details>
