{% test freshdata_expectation(model, column_name=none, rule_name='', message='', severity_label='') %}
{#-
    Placeholder generic test emitted by freshdata's quality-ops exporter
    (`freshdata.export_dbt_tests`) for *advanced* findings that have no native dbt
    generic test (regex / range / drift / privacy / entity-resolution checks).

    It does NOT re-run the original freshdata check — freshdata already evaluated it
    and recorded the result. The test exists to surface that finding inside the dbt
    DAG and `dbt test` output (carrying its rule_name, message, and severity), so
    quality-ops dashboards see one consistent set of tests. By default it returns no
    failing rows; override this macro in your project to re-assert the condition in
    SQL if you want dbt to enforce it independently.
-#}
select *
from {{ model }}
where false
{% endtest %}
