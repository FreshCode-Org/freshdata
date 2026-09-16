"""The shared adversarial trap corpus.

Four gold corpora already exist under ``benchmarks/`` (Gauntlet, TruthBench,
CleanBench and the ``benchmarks/fixtures`` generator).  They are authoritative
and this module does not replace them -- ``adapters.py`` re-exports their cases
through :class:`TrapCase` so they can be scored with one vocabulary.

What this module adds is the part none of them provide: a corpus that the ~272
ordinary files under ``tests/`` can import.  Today each test re-invents its own
trap literals, which is why ``'1,000'``, ``'1.000'``, ``'0001'`` and ``'None'``
appear as literals in zero test files.

The organising idea is the one the library's own design turns on: **the same
token means different things in different fields**.  Cases are therefore keyed
by ``(token, role)``, and the corpus deliberately contains the *same* token
under several roles with different expected dispositions.  A cleaner that
applies one rule per token -- rather than per token-in-context -- fails here.

Honesty rule
------------
An expected disposition is recorded only where the repository actually defines
one (documentation, an existing gold fixture, or an explicit code path).  Where
behaviour is genuinely undefined, ``expected`` is ``None`` and ``spec_gap``
says what decision is missing.  A ``None`` expectation is *measured and
reported*, never asserted -- inventing a contract here would be indistinguishable
from inventing one in the library.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from .dispositions import Disposition

__all__ = [
    "TrapCase",
    "TRAPS",
    "UNSET",
    "by_family",
    "by_role",
    "by_token",
    "families",
    "roles",
]

D = Disposition


class _Unset:
    """Sentinel: no gold repair value was supplied.

    Distinct from ``None``, which is a legitimate gold value meaning "the
    correct repair is to make this cell missing" (an ``NA`` sentinel in a free
    text column, say). Without this distinction those two cases collide.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "UNSET"

    def __bool__(self) -> bool:
        return False


UNSET = _Unset()


@dataclass(frozen=True)
class TrapCase:
    """One token, in one semantic role, with the disposition a human expects."""

    token: Any
    #: Trap family, e.g. ``"leading-zero-id"``. Shared with TruthBench naming.
    family: str
    #: The field the token sits in, e.g. ``"customer_id"``. Drives the context.
    role: str
    #: The semantic type hint a caller would declare for ``role``, if any.
    semantic_type: str | None
    #: What should happen. ``None`` means the repository does not define it.
    expected: Disposition | None
    #: Why -- in terms a reviewer can check, not a restatement of ``expected``.
    rationale: str
    #: The gold value when ``expected`` is ``REPAIR``. ``None`` means "repair
    #: to missing"; :data:`UNSET` means no value was supplied.
    repaired: Any = UNSET
    #: Where this case came from.
    source: str = "corpus"
    #: Set when ``expected`` is ``None``: the decision the repo has not made.
    spec_gap: str | None = None

    def __post_init__(self) -> None:
        if self.expected is None and not self.spec_gap:
            raise ValueError(
                f"{self.family}/{self.role}: an undefined expectation must carry "
                "a spec_gap explaining which decision is missing"
            )
        if self.expected is D.REPAIR and self.repaired is UNSET:
            raise ValueError(
                f"{self.family}/{self.role}: a REPAIR case must supply the gold "
                "'repaired' value, otherwise the repair cannot be scored"
            )
        if self.expected is not D.REPAIR and self.repaired is not UNSET:
            raise ValueError(
                f"{self.family}/{self.role}: 'repaired' is only meaningful for a "
                f"REPAIR case, got expected={self.expected}"
            )


def _c(*args: Any, **kw: Any) -> TrapCase:
    return TrapCase(*args, **kw)


# --------------------------------------------------------------------------
# 1. Identifier vs quantity -- the canonical same-token-different-role trap.
#    src/freshdata/semantic/experts.py:353 treats a >1-char all-digit string
#    starting with '0' as an identifier value; five independent layers protect
#    it. In a quantity column the numeric reading is the correct one.
# --------------------------------------------------------------------------
_IDENTITY = [
    _c(
        "007",
        "leading-zero-id",
        "customer_id",
        "identifier",
        D.PRESERVE,
        "Zero padding is payload in an identifier; dropping it merges 007 with 7.",
    ),
    _c(
        "007",
        "leading-zero-id",
        "postal_code",
        "postal_code",
        D.PRESERVE,
        "Postal codes are codes, not magnitudes; 007 and 7 are different places.",
    ),
    _c(
        "007",
        "leading-zero-id",
        "product_code",
        "category_code",
        D.PRESERVE,
        "Product codes are opaque tokens.",
    ),
    _c(
        "007",
        "leading-zero-numeric",
        "quantity",
        "quantity",
        D.REPAIR,
        "In a count column the value is the number seven; the padding is noise.",
        repaired=7,
    ),
    _c(
        "0001",
        "leading-zero-id",
        "invoice_no",
        "identifier",
        D.PRESERVE,
        "Same rule at a different width; guards against a 3-char special case.",
    ),
    _c(
        "0001",
        "leading-zero-numeric",
        "quantity",
        "quantity",
        D.REPAIR,
        "Numeric role: one.",
        repaired=1,
    ),
    _c(
        "0",
        "zero-is-valid",
        "quantity",
        "quantity",
        D.PRESERVE,
        "Zero is a legitimate count, not a missing value. The classic "
        "false-positive: treating 0 as a sentinel destroys real data.",
    ),
    _c(
        "0",
        "zero-is-valid",
        "account_id",
        "identifier",
        D.PRESERVE,
        "A single '0' is a valid identifier and must not become missing.",
    ),
    _c(
        "0.00",
        "free-item",
        "price",
        "currency_amount",
        D.PRESERVE,
        "A free item has a real price of zero; TruthBench family 'free-item'.",
        source="truthbench",
    ),
    _c(
        "A-100",
        "punctuated-code",
        "part_number",
        "category_code",
        D.PRESERVE,
        "Hyphenated codes are identifiers, not ranges or arithmetic.",
    ),
    _c(
        "BRK.B",
        "ticker-with-dot",
        "ticker",
        "category_code",
        D.PRESERVE,
        "A real share-class ticker; the dot is not a decimal point.",
    ),
    _c(
        "105A",
        "alphanumeric-id",
        "unit_number",
        "identifier",
        D.PRESERVE,
        "Mixed alphanumerics are identifiers (semantic/experts.py:353).",
    ),
]

# --------------------------------------------------------------------------
# 2. Sentinel-looking tokens that are sometimes real data.
#    fieldcheck.py:466-469 already documents that explicit vocabulary outranks
#    null markers -- 'NA' may be Namibia.
# --------------------------------------------------------------------------
_SENTINELS = [
    _c(
        "NA",
        "sentinel-vs-country",
        "country",
        "country",
        None,
        "ISO 3166-1 alpha-2 for Namibia, and also a default sentinel token.",
        spec_gap="The repository contradicts itself. fieldcheck.py:466 states "
        "'explicit vocabulary outranks generic null markers -- NA may be "
        "Namibia' and validate_fields honours a declared allowed_values; "
        "fd.clean's normalize_sentinels ignores allowed_values and nulls it "
        "(FD2-001). Gauntlet calls the collision contract behaviour. Until the "
        "two layers agree, the expectation is undefined.",
    ),
    _c(
        "NA",
        "sentinel-vs-surname",
        "last_name",
        "person_name",
        None,
        "'Na' is a real surname, and also a default sentinel token.",
        spec_gap="Same cross-layer conflict as country (FD2-001): a declared "
        "allowed_values protects the value in validate_fields but not in "
        "fd.clean. Undefined until the layers agree.",
    ),
    _c(
        "NA",
        "sentinel",
        "notes",
        "free_text",
        D.REPAIR,
        "In free text with no competing vocabulary, NA is a missing marker.",
        repaired=None,
    ),
    _c(
        "None",
        "sentinel-vs-brand",
        "product_name",
        "free_text",
        D.REPAIR,
        "'None' is a real brand/typeface name, but the repository has decided: "
        "Gauntlet labels this exact trap repair-to-missing and calls nulling it "
        "'contract behaviour' (gauntlet/fixtures.py:178). Encoded as the repo "
        "decided it, not as one might prefer -- the residual risk is recorded "
        "in FD2-001 instead.",
        repaired=None,
        source="gauntlet",
    ),
    _c(
        "None",
        "sentinel",
        "middle_name",
        "person_name",
        None,
        "Python's None repr and a deliberate 'no middle name' are indistinguishable "
        "from the token alone.",
        spec_gap="Undecided whether the literal string 'None' in a name column is "
        "a missing marker or asserted absence. Needs a documented rule.",
    ),
    _c(
        "null",
        "sentinel",
        "notes",
        "free_text",
        D.REPAIR,
        "Lowercase 'null' in free text is a missing marker.",
        repaired=None,
    ),
    _c(
        "",
        "empty-string",
        "notes",
        "free_text",
        D.REPAIR,
        "An empty string in free text is missing.",
        repaired=None,
    ),
    _c(
        "",
        "empty-string",
        "comment",
        "free_text",
        None,
        "An empty comment may mean 'deliberately blank'.",
        spec_gap="Undecided whether empty string is distinguishable from NULL for "
        "columns where blank is a meaningful user choice.",
    ),
]

# --------------------------------------------------------------------------
# 3. One-character category collisions. The same 'M' across four fields.
# --------------------------------------------------------------------------
_COLLISIONS = [
    _c(
        "M",
        "category-collision",
        "gender",
        "category_code",
        D.REPAIR,
        "In a gender field with a declared vocabulary, M expands to Male.",
        repaired="Male",
    ),
    _c(
        "M",
        "category-collision",
        "shirt_size",
        "category_code",
        D.REPAIR,
        "In a size field the same token expands to Medium -- the opposite of the "
        "gender expansion. This pair is the memory-replay trap.",
        repaired="Medium",
    ),
    _c(
        "M",
        "category-collision",
        "marital_status",
        "category_code",
        D.REPAIR,
        "And again to Married. One token, three correct answers.",
        repaired="Married",
    ),
    _c(
        "M",
        "category-collision",
        "initial",
        "person_name",
        D.PRESERVE,
        "A middle initial is already complete; expanding it fabricates data.",
    ),
    _c(
        "Male",
        "already-canonical",
        "gender",
        "category_code",
        D.PRESERVE,
        "Already the canonical form; a second pass must be a no-op (idempotency).",
    ),
]

# --------------------------------------------------------------------------
# 4. Spelled numbers. semantic/experts.py:71 invalidates the whole phrase on
#    any unknown token, which is what makes 'twenty apples' safe.
# --------------------------------------------------------------------------
_SPELLED = [
    _c(
        "twenty",
        "spelled-number",
        "quantity",
        "quantity",
        D.REPAIR,
        "Unambiguous spelled integer.",
        repaired=20,
    ),
    _c(
        "Twenty",
        "spelled-number-case",
        "quantity",
        "quantity",
        D.REPAIR,
        "Capitalisation is not semantic here.",
        repaired=20,
    ),
    _c(
        "twenty one",
        "spelled-number-compound",
        "quantity",
        "quantity",
        D.REPAIR,
        "Space-separated compound.",
        repaired=21,
    ),
    _c(
        "twenty-one",
        "spelled-number-hyphen",
        "quantity",
        "quantity",
        D.REPAIR,
        "Hyphenated compound must match the spaced form.",
        repaired=21,
    ),
    _c(
        "one hundred and twenty",
        "spelled-number-long",
        "quantity",
        "quantity",
        D.REPAIR,
        "British 'and' form.",
        repaired=120,
    ),
    _c(
        "twenty apples",
        "spelled-number-with-noun",
        "quantity",
        "quantity",
        D.REVIEW,
        "MUST NOT become 20. The cell carries a unit/noun the schema does not "
        "model, so the value is not a bare count. Silently dropping 'apples' "
        "is exactly the false-positive corruption this corpus exists to catch.",
    ),
    _c(
        "twenty",
        "spelled-number",
        "product_name",
        "free_text",
        D.PRESERVE,
        "In free text the word is the content.",
    ),
]

# --------------------------------------------------------------------------
# 5. Currency and decimal separators. semantic/experts.py:110 requires an
#    explicit marker, so a bare '1,200' is left to ordinary dtype repair.
# --------------------------------------------------------------------------
_CURRENCY = [
    _c(
        "$1,200.50",
        "currency-marked",
        "amount",
        "currency_amount",
        D.REPAIR,
        "Explicit symbol and unambiguous en-US grouping.",
        repaired=1200.50,
    ),
    _c(
        "1,200",
        "grouping-ambiguous",
        "amount",
        "currency_amount",
        None,
        "'1,200' is 1200 under en-US grouping and 1.2 under a decimal comma.",
        spec_gap="Undecided without a declared locale. The library must either "
        "require an explicit locale or route to review -- it must not "
        "silently assume en-US.",
    ),
    _c(
        "1.200",
        "grouping-ambiguous",
        "amount",
        "currency_amount",
        None,
        "'1.200' is 1.2 under en-US and 1200 under a decimal point as separator.",
        spec_gap="Undecided without a declared locale, and the mirror of "
        "'1,200': if the pair resolves under different conventions the "
        "same column yields 1200 and 1.2 for equivalent inputs.",
    ),
    _c(
        "EUR 1.200,50",
        "currency-declared-locale",
        "amount",
        "currency_amount",
        D.REPAIR,
        "Currency code plus a decimal comma is unambiguous.",
        repaired=1200.50,
    ),
    _c(
        "12.5%",
        "percent",
        "rate",
        "rate",
        D.REPAIR,
        "Percent literal; Gauntlet carries this trap.",
        repaired=0.125,
        source="gauntlet",
    ),
    _c(
        "$1,200.50",
        "currency-in-text",
        "description",
        "free_text",
        D.PRESERVE,
        "A price quoted inside prose is content, not a number to extract.",
    ),
]

# --------------------------------------------------------------------------
# 6. Units.
# --------------------------------------------------------------------------
_UNITS = [
    _c(
        "10 kg",
        "unit-matches-column",
        "weight_kg",
        "quantity_with_unit",
        D.REPAIR,
        "Unit agrees with the declared column unit.",
        repaired=10.0,
    ),
    _c(
        "10 kilograms",
        "unit-spelled",
        "weight_kg",
        "quantity_with_unit",
        D.REPAIR,
        "Spelled unit, same meaning.",
        repaired=10.0,
    ),
    _c(
        "10kg",
        "unit-no-space",
        "weight_kg",
        "quantity_with_unit",
        D.REPAIR,
        "Missing space is formatting, not meaning.",
        repaired=10.0,
    ),
    _c(
        "10 lbs",
        "unit-conflicts-column",
        "weight_kg",
        "quantity_with_unit",
        D.REVIEW,
        "The cell's unit contradicts the column's declared unit. Converting "
        "silently and stripping the unit both lose the conflict.",
    ),
    _c(
        "10",
        "unit-absent",
        "weight_kg",
        "quantity_with_unit",
        D.REPAIR,
        "Bare number in a column with a declared unit takes that unit.",
        repaired=10.0,
    ),
    _c(
        "10",
        "unit-unknown",
        "weight",
        None,
        None,
        "No declared column unit and no unit in the cell.",
        spec_gap="Undecided whether a bare number in an undeclared-unit column is "
        "accepted as-is or flagged. Must never be silently equated with "
        "values carrying a known unit.",
    ),
    _c(
        "10 zorkmids",
        "unit-unrecognised",
        "weight_kg",
        "quantity_with_unit",
        D.REVIEW,
        "An unknown unit must not be discarded to salvage the number.",
    ),
]

# --------------------------------------------------------------------------
# 7. Booleans. _TRUE_TOKENS/_FALSE_TOKENS at semantic/experts.py:159-160 are
#    deliberately narrow; 'enabled'/'active' are NOT in them.
# --------------------------------------------------------------------------
_BOOLEANS = [
    _c(
        "yes",
        "boolean-synonym",
        "is_active",
        "boolean_like",
        D.REPAIR,
        "In the documented true-token set.",
        repaired=True,
    ),
    _c(
        "Y",
        "boolean-synonym",
        "is_active",
        "boolean_like",
        D.REPAIR,
        "In the documented true-token set.",
        repaired=True,
    ),
    _c(
        "TRUE",
        "boolean-synonym",
        "is_active",
        "boolean_like",
        D.REPAIR,
        "In the documented true-token set.",
        repaired=True,
    ),
    _c(
        "1",
        "boolean-numeric",
        "is_active",
        "boolean_like",
        D.REPAIR,
        "Declared boolean column: 1 is true.",
        repaired=True,
    ),
    _c(
        "1",
        "boolean-numeric",
        "login_count",
        "quantity",
        D.PRESERVE,
        "Same token in a count column is the number one, not True. "
        "semantic/context.py:166 never sets boolean_like for a numeric role.",
    ),
    _c(
        "enabled",
        "boolean-correlated-word",
        "status",
        None,
        None,
        "'enabled' correlates with true but is not in the documented token set, "
        "and a status column may have more than two states.",
        spec_gap="Undecided whether status vocabularies collapse to boolean. "
        "Must not assume every truthy-sounding word is True.",
    ),
    _c(
        "active",
        "boolean-correlated-word",
        "status",
        None,
        None,
        "Same: 'active' may sit beside 'suspended' and 'pending'.",
        spec_gap="Undecided whether a status vocabulary collapses to boolean. "
        "'active' commonly sits beside 'suspended', 'pending' and "
        "'cancelled', so a two-valued reading silently discards states.",
    ),
    _c(
        "unknown",
        "boolean-third-state",
        "is_active",
        "boolean_like",
        D.REVIEW,
        "'unknown' is a third state, not False. Mapping it to False fabricates a negative answer.",
    ),
    _c(
        "N/A",
        "boolean-missing",
        "is_active",
        "boolean_like",
        D.REPAIR,
        "Not-applicable in a boolean column is missing, not False.",
        repaired=None,
    ),
]

# --------------------------------------------------------------------------
# 8. Dates.
# --------------------------------------------------------------------------
_DATES = [
    _c(
        "01/02/2026",
        "date-ambiguous",
        "order_date",
        "date_like",
        None,
        "1 Feb or 2 Jan depending on convention; both are real dates, so no "
        "parse error reveals the mistake.",
        spec_gap="Undecided without dayfirst. Must route to review rather than "
        "silently pick a convention (experts.py:291 marks these "
        "high-risk; the corpus verifies that end to end).",
    ),
    _c(
        "02/01/2026",
        "date-ambiguous",
        "order_date",
        "date_like",
        None,
        "The mirror case. If the pair resolves inconsistently, ordering breaks.",
        spec_gap="Undecided without dayfirst, and critically the mirror of "
        "'01/02/2026': both must resolve under the SAME convention or "
        "date ordering between the two rows silently inverts.",
    ),
    _c(
        "2026-02-01",
        "date-iso",
        "order_date",
        "date_like",
        D.REPAIR,
        "ISO 8601 is unambiguous by construction.",
        repaired="2026-02-01",
    ),
    _c(
        "2026-13-01",
        "date-impossible",
        "order_date",
        "date_like",
        D.QUARANTINE,
        "Month 13 cannot be repaired without guessing which field is wrong.",
    ),
    _c(
        "2023-02-30",
        "date-impossible",
        "order_date",
        "date_like",
        D.QUARANTINE,
        "February 30 never existed; Gauntlet carries this trap.",
        source="gauntlet",
    ),
    _c(
        "today",
        "date-relative",
        "order_date",
        "date_like",
        None,
        "Resolvable only against a reference date.",
        spec_gap="With an explicit reference_date the result must be deterministic "
        "and independent of the real clock; without one the library must "
        "not silently use today's date.",
    ),
]

# --------------------------------------------------------------------------
# 9. Valid-but-weird entities. These exist to measure false-positive
#    corruption -- every one is real data that a naive cleaner damages.
# --------------------------------------------------------------------------
_ENTITIES = [
    _c(
        "O'Connor",
        "apostrophe-name",
        "last_name",
        "person_name",
        D.PRESERVE,
        "The apostrophe is part of the name. Stripping punctuation, or escaping "
        "it as if it were SQL, corrupts a very common surname.",
    ),
    _c(
        "X Æ A-12",
        "unusual-name",
        "first_name",
        "person_name",
        D.PRESERVE,
        "A real registered given name mixing a ligature, spaces and a hyphen-digit.",
    ),
    _c(
        "José",
        "accented-name",
        "first_name",
        "person_name",
        D.PRESERVE,
        "Accents are content. NFC normalisation may change the byte sequence but "
        "must not change the rendered name.",
    ),
    _c(
        "john@example.com",
        "email-valid",
        "email",
        "email",
        D.PRESERVE,
        "Already valid; must not be 'repaired'.",
    ),
    _c(
        "John &amp; Sons",
        "html-entity",
        "company_name",
        "free_text",
        D.REPAIR,
        "A real ampersand that survived an HTML round-trip.",
        repaired="John & Sons",
    ),
    _c(
        "<b>ABC</b>",
        "html-markup",
        "company_name",
        "free_text",
        D.REPAIR,
        "Presentational markup around the real value.",
        repaired="ABC",
    ),
    _c(
        "a < b and c > d",
        "html-lookalike",
        "notes",
        "free_text",
        D.PRESERVE,
        "Angle brackets used as comparison operators are not markup.",
    ),
    _c(
        "茶",
        "multilingual",
        "product_name",
        "free_text",
        D.PRESERVE,
        "Non-Latin scripts are content; TruthBench 'multilingual-product'.",
        source="truthbench",
    ),
]

# --------------------------------------------------------------------------
# 10. Near misses -- values one character away from valid. These exist to
#     measure false negatives: a regex-only validator accepts them.
# --------------------------------------------------------------------------
_NEAR_MISSES = [
    _c(
        "100O",
        "ocr-digit-letter",
        "quantity",
        "quantity",
        D.REVIEW,
        "Trailing capital O for a zero. Coercing to 100 guesses; rejecting "
        "outright loses a recoverable value. A human must choose.",
    ),
    _c(
        "l0",
        "ocr-letter-digit",
        "quantity",
        "quantity",
        D.REVIEW,
        "Lowercase L for a one. Same argument.",
    ),
    _c(
        "1,2O0.50",
        "ocr-in-currency",
        "amount",
        "currency_amount",
        D.REVIEW,
        "A letter inside an otherwise well-formed money string.",
    ),
    _c(
        "AAPL ",
        "trailing-space-ticker",
        "ticker",
        "category_code",
        D.REPAIR,
        "Trailing whitespace is never payload in a ticker.",
        repaired="AAPL",
    ),
    _c(
        "Malee",
        "near-miss-name",
        "first_name",
        "person_name",
        D.PRESERVE,
        "One letter from 'Malee'/'Marlee' but itself a real name. Fuzzy-matching "
        "it to a reference list is false-positive corruption.",
    ),
    _c(
        "not-an-email@",
        "email-malformed",
        "email",
        "email",
        D.QUARANTINE,
        "Structurally invalid; recoverable for review but not usable.",
    ),
    _c(
        "test@test.test",
        "email-plausible-fake",
        "email",
        "email",
        D.FLAG,
        "Syntactically valid and semantically worthless. Regex validation alone "
        "cannot catch it, which is the point.",
    ),
]

# --------------------------------------------------------------------------
# 11. Unicode traps. textclean strips zero-width by default; there is no
#     homoglyph detection in src/ (gap B5), so those cases carry a spec_gap.
# --------------------------------------------------------------------------
_UNICODE = [
    _c(
        "A\u200bpple",
        "zero-width",
        "company_name",
        "free_text",
        D.REPAIR,
        "A zero-width space inside a word is invisible and breaks equality "
        "against the clean form.",
        repaired="Apple",
    ),
    _c(
        "Аpple",
        "homoglyph-cyrillic",
        "company_name",
        "free_text",
        None,
        "Leading character is Cyrillic A, not Latin A. Renders identically.",
        spec_gap="No homoglyph/confusable detection exists in src/ (nearest is "
        "textlint mixed_script, diagnostic only). Undecided whether this "
        "is flagged, repaired or preserved.",
    ),
    _c(
        "ＡＰＰＬ",
        "fullwidth",
        "ticker",
        "category_code",
        None,
        "Full-width Latin letters spelling AAPL.",
        spec_gap="NFKC would fold these to ASCII but the semantic repair path uses "
        "NFC only (semantic/canonical.py:71). Undecided for identifier-like "
        "columns where folding changes the payload.",
    ),
    _c(
        "José",
        "combining-mark",
        "first_name",
        "person_name",
        D.REPAIR,
        "Decomposed e-acute; NFC composes it without changing the rendered name.",
        repaired="José",
    ),
    _c(
        "CafÃ©",
        "mojibake",
        "company_name",
        "free_text",
        D.REPAIR,
        "UTF-8 bytes read as Latin-1; TruthBench 'html-entity-mojibake'.",
        repaired="Café",
        source="truthbench",
    ),
]

# --------------------------------------------------------------------------
# 12. Values that are valid in one column and invalid in another -- the
#     field-validation matrix from Phase 7.
# --------------------------------------------------------------------------
_FIELD_MATRIX = [
    _c(
        "apple",
        "text-in-numeric",
        "amount",
        "currency_amount",
        D.QUARANTINE,
        "Unparseable as money; fieldcheck defaults parse_failure to quarantine "
        "so the original stays recoverable.",
    ),
    _c(
        "apple",
        "text-in-text",
        "company_name",
        "free_text",
        D.PRESERVE,
        "A perfectly good company name.",
    ),
    _c(
        "apple",
        "text-in-ticker",
        "ticker",
        "category_code",
        D.QUARANTINE,
        "Not a valid ticker; the real one is AAPL.",
    ),
    _c(
        "apple",
        "text-in-freetext",
        "notes",
        "free_text",
        D.PRESERVE,
        "Free text accepts any string.",
    ),
]

TRAPS: tuple[TrapCase, ...] = tuple(
    _IDENTITY
    + _SENTINELS
    + _COLLISIONS
    + _SPELLED
    + _CURRENCY
    + _UNITS
    + _BOOLEANS
    + _DATES
    + _ENTITIES
    + _NEAR_MISSES
    + _UNICODE
    + _FIELD_MATRIX
)


def by_family(family: str) -> tuple[TrapCase, ...]:
    """Every case in one trap family."""
    return tuple(c for c in TRAPS if c.family == family)


def by_role(role: str) -> tuple[TrapCase, ...]:
    """Every case that sits in one field."""
    return tuple(c for c in TRAPS if c.role == role)


def by_token(token: Any) -> tuple[TrapCase, ...]:
    """Every role a given token appears in -- the same-token-different-role view."""
    return tuple(c for c in TRAPS if c.token == token)


def families() -> tuple[str, ...]:
    return tuple(sorted({c.family for c in TRAPS}))


def roles() -> tuple[str, ...]:
    return tuple(sorted({c.role for c in TRAPS}))


def __iter__() -> Iterator[TrapCase]:  # pragma: no cover - convenience only
    return iter(TRAPS)
