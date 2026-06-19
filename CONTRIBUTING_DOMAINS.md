# Contributing a domain validator pack

A *domain pack* teaches `fd.clean(df, domain="…")` how to validate and repair a
specific kind of tabular data against versioned, reviewable rules. Built-in packs
live in `src/freshdata/domains/<name>/`; third-party packs ship as separate
PyPI distributions and register through an entry point. Both implement the same
interface.

## The interface

Every pack implements `freshdata.domains.DomainValidator`:

```python
class DomainValidator(ABC):
    domain_name: str        # "finance"
    version: str            # "0.1.0"
    schema_version: str     # "2024-01"

    def detect_columns(self, df) -> ColumnMapping: ...
    def validate(self, df) -> ValidationReport: ...
    def repair(self, df, report) -> tuple[DataFrame, RepairLog]: ...
    def describe(self) -> dict: ...
```

Most packs subclass **`ConfigDrivenValidator`**, which implements the layered
engine for you — you supply a `rules.yaml`, declare canonical fields, and add any
custom checks. The finance pack (`src/freshdata/domains/finance/`) is the
reference example.

## Anatomy of a pack

```
mypack/
  __init__.py        # exports your Validator class
  validator.py       # subclass ConfigDrivenValidator; register custom checks
  rules.yaml         # the rules (see below)
  reference/*.json   # bundled code sets, each with a _meta block
```

### `validator.py`

```python
from freshdata.domains import ConfigDrivenValidator

class MyValidator(ConfigDrivenValidator):
    domain_name = "mydomain"
    version = "0.1.0"
    schema_version = "2024-01"

    canonical_fields = ("id", "amount", "code")
    required_fields = ("id", "amount")
    id_fields = ("id",)                      # never repaired
    aliases = {"id": (r"my_?id", r"identifier")}   # regex, case-insensitive
    rules_path = str(Path(__file__).parent / "rules.yaml")

    def register_extensions(self):
        self.register_check("my_check", self._my_check)        # params.func
        self.register_repair("my_fix", self._my_fix)           # repair_params.func

    def load_reference_values(self, name):                     # for reference checks
        ...
```

### `rules.yaml`

Each rule carries `id`, `name`, `layer` (`schema|format|reference|business|
semantic`), `severity` (`error|warning|info`), `field`/`fields`, `check`
(`not_null|required|regex|enum|reference|range|custom`), optional `params`, and
an optional `repair` (`fill_default|coerce|flag_only|reject|none`) with
`repair_params`. Layers always run in order; a rule whose target column is
absent is skipped (the schema layer reports it as `MISSING_REQUIRED_FIELD`).

### Reference data

Bundle code sets as static JSON with a `_meta` block so they stay auditable:

```json
{ "_meta": {"source": "ISO 4217", "retrieved_date": "2024-01-15", "version": "2024-01"},
  "codes": ["USD", "EUR", "..."] }
```

## Rules of the road

- **Validation never mutates data.** All changes happen in `repair`, and every
  attempt is logged to the `RepairLog` with from/to values, the rule id, and a
  status (`applied`/`flagged`/`unresolvable`).
- **Identifier columns are never repaired** — list them in `id_fields`.
- **Never guess column mappings silently** — detection logs how each match was made.
- **No network calls or LLM calls** in a pack.

## Registering a third-party pack

Expose your validator through the `freshdata.domains` entry-point group in your
package's `pyproject.toml`:

```toml
[project.entry-points."freshdata.domains"]
mydomain = "mypack.validator:MyValidator"
```

Once installed, `fd.clean(df, domain="mydomain")` finds it automatically.
Built-in names take precedence, so you cannot shadow `finance` (etc.).
