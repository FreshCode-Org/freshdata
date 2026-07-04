# Model cards

Every optional FreshData model artifact ships with a model card covering:
purpose, non-goals, runtime behavior, training data sources, license
summary, PII policy, teacher-model usage, human-review process, evaluation
metrics, known limitations, failure modes, the protected-column guarantee,
calibration version, artifact hash, and FreshData version compatibility.

The authoritative cards live in the repository and are copied into each
packaged artifact (`dist/artifacts/<model_id>/model_card.md`):

- [`fd-col-encoder-v1`](https://github.com/FreshCode-Org/freshdata/blob/main/training/model_cards/fd-col-encoder-v1.md)
  — column/value encoder + semantic-type role head
- [`fd-intent-v1`](https://github.com/FreshCode-Org/freshdata/blob/main/training/model_cards/fd-intent-v1.md)
  — context-sentence intent head (optional evidence for the deterministic parser)
- [`calib-v1`](https://github.com/FreshCode-Org/freshdata/blob/main/training/model_cards/calib-v1.md)
  — isotonic confidence-calibration tables (plain JSON)

Every card states explicitly:

- there is **no generative repair model**
- there is **no runtime LLM**
- there is **no automatic model download** during cleaning
- model output is **evidence, not authority** — the policy gate decides
- the **protected-column guard remains absolute**

Install status and hashes are always inspectable locally:

```python
import freshdata as fd
fd.models.status()
```
