# Security Policy

We take the security of **freshdata** seriously. This document explains which
versions receive security fixes and how to report a vulnerability responsibly.

## Supported Versions

Security fixes are applied to the latest released version. We generally do not
backport fixes to older releases — please upgrade to the latest.

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Instead, report privately using either of the following:

1. **GitHub Security Advisories** (preferred) — open a private report at
   <https://github.com/FreshCode-Org/freshdata/security/advisories/new>.
2. **Email** — contact the maintainer at **jyothiswaroop2803@gmail.com** with
   the subject line `freshdata security`.

Please include:

- A description of the vulnerability and its impact.
- Steps to reproduce (a minimal `pandas`/`freshdata` snippet is ideal).
- The `freshdata` version and Python version affected.

## What to Expect

- **Acknowledgement** within 5 business days.
- An initial assessment and severity classification within 10 business days.
- Coordinated disclosure: we will agree on a timeline before any public
  details are released, and credit reporters who wish to be named.

## Scope

`freshdata` processes user-supplied tabular data in-memory. Reports of memory
exhaustion on adversarial inputs, unsafe deserialization, or code execution via
crafted data files are in scope. General data-quality bugs should be filed as
regular GitHub issues instead.
