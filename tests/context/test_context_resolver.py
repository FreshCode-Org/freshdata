"""Resolver ladder: exact, snake_case, alias, token, difflib, ambiguity."""

from freshdata.context import resolve_reference

SCHEMA = ["cust_id", "full_name", "email_addr", "mobile", "age", "monthly_revenue", "city"]


def test_exact_match():
    r = resolve_reference("age", SCHEMA)
    assert (r.column, r.method, r.confidence) == ("age", "exact", 1.0)


def test_snake_case_match():
    r = resolve_reference("Age", SCHEMA)
    assert (r.column, r.method, r.confidence) == ("age", "normalized", 1.0)
    r = resolve_reference("Full Name", SCHEMA)
    assert (r.column, r.method) == ("full_name", "normalized")


def test_alias_match():
    r = resolve_reference("CustomerID", SCHEMA)
    assert (r.column, r.method) == ("cust_id", "alias")
    assert r.confidence == 0.90
    assert resolve_reference("Emails", SCHEMA).column == "email_addr"
    assert resolve_reference("Phone numbers", SCHEMA).column == "mobile"
    assert resolve_reference("revenue", SCHEMA).column == "monthly_revenue"


def test_token_subset_match():
    schema = ["order_total_amount_usd", "city"]
    r = resolve_reference("order total", schema)
    assert (r.column, r.method) == ("order_total_amount_usd", "tokens")


def test_difflib_match():
    schema = ["custmer_email", "city"]  # misspelled column
    r = resolve_reference("customer email", schema)
    assert (r.column, r.method) == ("custmer_email", "difflib")
    assert r.confidence >= 0.85


def test_alias_ambiguity_is_unresolved():
    schema = ["phone", "mobile", "age"]  # two members of the phone alias group
    r = resolve_reference("Phone numbers", schema)
    assert r.column is None
    assert r.method == "unresolved"
    assert {c for c, _ in r.candidates} == {"phone", "mobile"}


def test_token_tie_is_unresolved():
    schema = ["ship_date_1", "ship_date_2"]
    r = resolve_reference("ship date", schema)
    assert r.column is None
    assert {c for c, _ in r.candidates} == set(schema)


def test_difflib_near_tie_is_unresolved():
    schema = ["email_addr", "email_addrs"]  # both fuzzy-close to the typo
    r = resolve_reference("email_adr", schema)
    assert r.column is None
    assert "ambiguous" in r.reason


def test_no_match_is_unresolved_with_candidates():
    r = resolve_reference("blood_pressure", SCHEMA)
    assert r.column is None
    assert r.method == "unresolved"
    assert len(r.candidates) == 3  # top-3 shortlist, best first
    assert r.candidates == tuple(sorted(r.candidates, key=lambda p: -p[1]))


def test_resolver_never_invents_columns():
    r = resolve_reference("anything", [])
    assert r.column is None


def test_resolver_deterministic():
    assert resolve_reference("Emails", SCHEMA) == resolve_reference("Emails", SCHEMA)
