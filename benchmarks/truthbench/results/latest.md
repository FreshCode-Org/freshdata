# TruthBench run

- run id: `tb-c07b1bf21cbe`
- profile: `release`
- records: 3144
- required backends: pandas, polars, duckdb
- gates: 41/48 passed
- overall: FAIL

## Environment

- duckdb: `1.5.4`
- freshdata: `1.1.1`
- numpy: `2.4.6`
- pandas: `2.3.3`
- platform: `macOS-15.5-arm64-arm-64bit-Mach-O`
- polars: `1.42.1`
- python: `3.13.9`

## Gates

| gate | status | failures |
|---|---|---|
| cleaning:completeness | ✅ pass | 0 |
| cleaning:schema_validation | ✅ pass | 0 |
| cleaning:fixture_evidence | ✅ pass | 0 |
| cleaning:unexpected_exception | ✅ pass | 0 |
| cleaning:required_backend | ✅ pass | 0 |
| cleaning:valid_value_corruption | ❌ FAIL | 336 |
| cleaning:protected_column_modification | ✅ pass | 0 |
| cleaning:raw_pii_leakage | ❌ FAIL | 96 |
| cleaning:backend_inconsistency | ✅ pass | 0 |
| cleaning:default_nondeterminism | ✅ pass | 0 |
| cleaning:broken_generated_code | ✅ pass | 0 |
| cleaning:unexplained_high_confidence | ✅ pass | 0 |
| cleaning:trust_inversion | ✅ pass | 0 |
| cleaning:mutation_audit | ✅ pass | 0 |
| cleaning:review_routing | ❌ FAIL | 70 |
| cleaning:exact_repair | ❌ FAIL | 24 |
| cleaning:flag_mutation | ❌ FAIL | 14 |
| cleaning:validator_contract | ✅ pass | 0 |
| cleaning:pii_scope | ✅ pass | 0 |
| cleaning:requested_behavior | ✅ pass | 0 |
| cleaning:input_mutation | ✅ pass | 0 |
| cleaning:case_coverage | ❌ FAIL | 1 |
| cleaning:aggregate_consistency | ✅ pass | 0 |
| parity:completeness | ✅ pass | 0 |
| parity:schema_validation | ✅ pass | 0 |
| parity:fixture_evidence | ✅ pass | 0 |
| parity:unexpected_exception | ✅ pass | 0 |
| parity:required_backend | ✅ pass | 0 |
| parity:valid_value_corruption | ✅ pass | 0 |
| parity:protected_column_modification | ✅ pass | 0 |
| parity:raw_pii_leakage | ✅ pass | 0 |
| parity:backend_inconsistency | ✅ pass | 0 |
| parity:default_nondeterminism | ✅ pass | 0 |
| parity:broken_generated_code | ✅ pass | 0 |
| parity:unexplained_high_confidence | ✅ pass | 0 |
| parity:trust_inversion | ✅ pass | 0 |
| parity:mutation_audit | ✅ pass | 0 |
| parity:review_routing | ✅ pass | 0 |
| parity:exact_repair | ✅ pass | 0 |
| parity:flag_mutation | ✅ pass | 0 |
| parity:validator_contract | ✅ pass | 0 |
| parity:pii_scope | ✅ pass | 0 |
| parity:requested_behavior | ✅ pass | 0 |
| parity:input_mutation | ✅ pass | 0 |
| parity:case_coverage | ✅ pass | 0 |
| parity:aggregate_consistency | ✅ pass | 0 |
| backend_parity_comparison | ✅ pass | 0 |
| generated_code_sandbox | ❌ FAIL | 6 |

## Failure detail

### cleaning:valid_value_corruption

- tb-c07b1bf21cbe:cleaning:pandas:0:v1:crm:crm-01:signup_date: preserve value was corrupted
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:crm:crm-01:signup_date: preserve value was corrupted
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:crm:crm-02:signup_date: preserve value was corrupted
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:crm:crm-02:signup_date: preserve value was corrupted
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:crm:crm-03:signup_date: preserve value was corrupted
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:crm:crm-03:signup_date: preserve value was corrupted
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:crm:crm-04:signup_date: preserve value was corrupted
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:crm:crm-04:signup_date: preserve value was corrupted
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:crm:crm-05:signup_date: preserve value was corrupted
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:crm:crm-05:signup_date: preserve value was corrupted
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:crm:crm-06:signup_date: preserve value was corrupted
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:crm:crm-06:signup_date: preserve value was corrupted
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:crm:crm-08:signup_date: preserve value was corrupted
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:crm:crm-08:signup_date: preserve value was corrupted
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:crm:crm-09:signup_date: preserve value was corrupted
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:crm:crm-09:signup_date: preserve value was corrupted
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:crm:crm-10:signup_date: preserve value was corrupted
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:crm:crm-10:signup_date: preserve value was corrupted
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:crm:crm-11:signup_date: preserve value was corrupted
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:crm:crm-11:signup_date: preserve value was corrupted
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:crm:crm-12:signup_date: preserve value was corrupted
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:crm:crm-12:signup_date: preserve value was corrupted
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:crm:crm-13:signup_date: preserve value was corrupted
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:crm:crm-13:signup_date: preserve value was corrupted
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:crm:crm-14:signup_date: preserve value was corrupted
- … and 311 more

### cleaning:raw_pii_leakage

- sink[0] leaked canary crm-crm-09-email at $.report.actions[2].description
- sink[0] leaked canary crm-crm-09-email at $.report.actions[2].rationale
- sink[0] leaked canary crm-crm-09-email at $.report.actions[2].metadata.raw_value
- sink[0] leaked canary crm-crm-09-email at $.report.actions[2].metadata.evidence[0].detail
- sink[13] leaked canary crm-crm-10-notes at $.rendered.to_dict.stakeholder.headline
- sink[13] leaked canary crm-crm-10-notes at $.rendered.json
- sink[13] leaked canary crm-crm-10-notes at $.rendered.markdown
- sink[13] leaked canary crm-crm-10-notes at $.rendered.html
- sink[13] leaked canary crm-crm-10-notes at $.debt_output.account_id['fin-01']
- sink[14] leaked canary crm-crm-10-notes at $.rendered.json
- sink[14] leaked canary crm-crm-10-notes at $.rendered.html
- sink[38] leaked canary crm-crm-10-notes at $.rendered.to_dict.stakeholder.headline
- sink[38] leaked canary crm-crm-10-notes at $.rendered.json
- sink[38] leaked canary crm-crm-10-notes at $.rendered.markdown
- sink[38] leaked canary crm-crm-10-notes at $.rendered.html
- sink[38] leaked canary crm-crm-10-notes at $.debt_output.sku['ret-05']
- sink[38] leaked canary crm-crm-10-notes at $.debt_output.sku['ret-14']
- sink[38] leaked canary crm-crm-10-notes at $.debt_output.sku['ret-15']
- sink[38] leaked canary crm-crm-10-notes at $.debt_output.sku['ret-16']
- sink[38] leaked canary crm-crm-10-notes at $.debt_output.gtin['ret-01']
- sink[38] leaked canary crm-crm-10-notes at $.debt_output.gtin['ret-02']
- sink[38] leaked canary crm-crm-10-notes at $.debt_output.gtin['ret-05']
- sink[38] leaked canary crm-crm-10-notes at $.debt_output.gtin['ret-14']
- sink[38] leaked canary crm-crm-10-notes at $.debt_output.gtin['ret-15']
- sink[38] leaked canary crm-crm-10-notes at $.debt_output.gtin['ret-16']
- … and 71 more

### cleaning:review_routing

- tb-c07b1bf21cbe:cleaning:pandas:0:v1:crm:crm-05:country: review case was not routed to a human
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:crm:crm-05:country: review case was not routed to a human
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:crm:crm-06:language: review case was not routed to a human
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:crm:crm-06:language: review case was not routed to a human
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:crm:crm-07:signup_date: review case was not routed to a human
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:crm:crm-07:signup_date: review case was not routed to a human
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:crm:crm-08:lifecycle: review case was not routed to a human
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:crm:crm-08:lifecycle: review case was not routed to a human
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:crm:crm-16:customer_id: review case was not routed to a human
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:crm:crm-16:customer_id: review case was not routed to a human
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:education:edu-04:school_year: review case was not routed to a human
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:education:edu-04:school_year: review case was not routed to a human
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:education:edu-05:enrollment_date: review case was not routed to a human
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:education:edu-05:enrollment_date: review case was not routed to a human
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:education:edu-06:completion_date: review case was not routed to a human
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:education:edu-06:completion_date: review case was not routed to a human
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:education:edu-16:grade_letter: review case was not routed to a human
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:education:edu-16:grade_letter: review case was not routed to a human
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:finance:fin-03:price: review case was not routed to a human
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:finance:fin-03:price: review case was not routed to a human
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:finance:fin-04:ticker: review case was not routed to a human
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:finance:fin-04:ticker: review case was not routed to a human
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:finance:fin-08:currency: review case was not routed to a human
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:finance:fin-08:currency: review case was not routed to a human
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:finance:fin-09:currency: review case was not routed to a human
- … and 45 more

### cleaning:exact_repair

- tb-c07b1bf21cbe:cleaning:pandas:0:v1:crm:crm-02:first_name: repair differs from the exact oracle
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:crm:crm-02:first_name: repair differs from the exact oracle
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:crm:crm-04:phone: repair differs from the exact oracle
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:crm:crm-04:phone: repair differs from the exact oracle
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:education:edu-07:score_percent: repair differs from the exact oracle
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:education:edu-07:score_percent: repair differs from the exact oracle
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:finance:fin-07:price: repair differs from the exact oracle
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:finance:fin-07:price: repair differs from the exact oracle
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:government:gov-06:encoding: repair differs from the exact oracle
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:government:gov-06:encoding: repair differs from the exact oracle
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:healthcare:hc-04:dose: repair differs from the exact oracle
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:healthcare:hc-04:dose: repair differs from the exact oracle
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:healthcare:hc-06:event_date: repair differs from the exact oracle
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:healthcare:hc-06:event_date: repair differs from the exact oracle
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:healthcare:hc-08:patient_name: repair differs from the exact oracle
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:healthcare:hc-08:patient_name: repair differs from the exact oracle
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:insurance:ins-03:premium: repair differs from the exact oracle
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:insurance:ins-03:premium: repair differs from the exact oracle
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:logistics:log-02:weight: repair differs from the exact oracle
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:logistics:log-02:weight: repair differs from the exact oracle
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:logistics:log-06:transport_time: repair differs from the exact oracle
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:logistics:log-06:transport_time: repair differs from the exact oracle
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:retail:ret-03:price: repair differs from the exact oracle
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:retail:ret-03:price: repair differs from the exact oracle

### cleaning:flag_mutation

- tb-c07b1bf21cbe:cleaning:pandas:0:v1:crm:crm-07:signup_date: flag/review value was mutated
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:crm:crm-07:signup_date: flag/review value was mutated
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:education:edu-05:enrollment_date: flag/review value was mutated
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:education:edu-05:enrollment_date: flag/review value was mutated
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:education:edu-06:completion_date: flag/review value was mutated
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:education:edu-06:completion_date: flag/review value was mutated
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:finance:fin-10:trade_date: flag/review value was mutated
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:finance:fin-10:trade_date: flag/review value was mutated
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:insurance:ins-06:report_date: flag/review value was mutated
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:insurance:ins-06:report_date: flag/review value was mutated
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:logistics:log-05:delivery_window: flag/review value was mutated
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:logistics:log-05:delivery_window: flag/review value was mutated
- tb-c07b1bf21cbe:cleaning:pandas:0:v1:retail:ret-04:price: flag/review value was mutated
- tb-c07b1bf21cbe:cleaning:pandas:1:v1:retail:ret-04:price: flag/review value was mutated

### cleaning:case_coverage

- required case v1:education:schema:type-drifted-score was not observed

### generated_code_sandbox

- stdout leaked canary crm-crm-10-notes at $
- stdout leaked canary education-edu-10-ferpa_notes at $
- stdout leaked canary government-gov-09-notes at $
- stdout leaked canary healthcare-hc-09-notes at $
- stdout leaked canary insurance-ins-09-claimant_name at $
- stdout leaked canary logistics-log-09-address at $


## Minimized failures

- `tbf-87a8962c331de1c5` valid_value_corruption @ v1:crm:crm-01:signup_date (1 rows) — `PYTHONPATH=src python -m benchmarks.truthbench reproduce --failure-id tbf-87a8962c331de1c5 --domain crm --cell-id 'v1:crm:crm-01:signup_date' --surface cleaning`
- `tbf-87a8962c331de1c5` valid_value_corruption @ v1:crm:crm-01:signup_date (1 rows) — `PYTHONPATH=src python -m benchmarks.truthbench reproduce --failure-id tbf-87a8962c331de1c5 --domain crm --cell-id 'v1:crm:crm-01:signup_date' --surface cleaning`
- `tbf-013fa51d68728a5e` valid_value_corruption @ v1:crm:crm-02:signup_date (1 rows) — `PYTHONPATH=src python -m benchmarks.truthbench reproduce --failure-id tbf-013fa51d68728a5e --domain crm --cell-id 'v1:crm:crm-02:signup_date' --surface cleaning`
