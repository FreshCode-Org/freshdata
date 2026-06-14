---
name: Bug report
about: Create a report to help us improve freshdata
title: '[BUG] '
labels: bug
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior. Please provide a minimal self-contained Python script with a sample DataFrame if possible:
```python
import pandas as pd
import freshdata as fd

df = pd.DataFrame(...)
fd.clean(df)
```

**Expected behavior**
A clear and concise description of what you expected to happen.

**Profile/Report Payload (if applicable)**
If you can, please include the output of `fd.profile(df)` or `report.to_dict()`.

**Environment (please complete the following information):**
- OS: [e.g. macOS, Ubuntu, Windows]
- Python Version [e.g. 3.9, 3.12]
- Pandas Version [e.g. 2.2.0, 3.0.3]
- freshdata-cleaner Version [e.g. 0.5.0]
