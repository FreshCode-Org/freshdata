# Model cards

Every optional FreshData model artifact ships with a model card covering:
purpose, non-goals, runtime behavior, training data sources, license
summary, PII policy, teacher-model usage, human-review process, evaluation
metrics, known limitations, failure modes, the protected-column guarantee,
calibration version, artifact hash, and FreshData version compatibility.

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
