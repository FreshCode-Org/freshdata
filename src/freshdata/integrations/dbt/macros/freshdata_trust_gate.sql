{#-
  freshdata_trust_gate — document/log the freshdata trust-gate invocation for a dbt run.

  freshdata gates data quality in Python, which dbt cannot execute inline, so the
  gate runs as the `dbt-gate` console script (pip install "freshdata[dbt]") against
  the manifest dbt just wrote. Wire this macro into on-run-end to emit the exact
  command for your CI step:

      # dbt_project.yml
      on-run-end:
        - "{{ freshdata_trust_gate(threshold=80) }}"

  Then run, after `dbt run`:

      dbt-gate --manifest {{ target.path }}/manifest.json --threshold 80 --fail
-#}
{% macro freshdata_trust_gate(threshold=80) %}
  {% if execute %}
    {% set manifest_path = target.path ~ '/manifest.json' %}
    {% set command = 'dbt-gate --manifest ' ~ manifest_path ~ ' --threshold ' ~ threshold ~ ' --fail' %}
    {% do log('freshdata trust gate — run as a CI step: ' ~ command, info=True) %}
  {% endif %}
{% endmacro %}
