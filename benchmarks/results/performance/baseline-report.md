# FreshData performance evidence

## Environment

- cpu_count_logical: `8`
- cpu_count_physical: `8`
- freshdata_version: `1.1.1`
- git_commit: `27ee7b653a1318ac66692df2ba9c7813a674e8aa`
- git_dirty: `True`
- numpy_version: `2.4.6`
- optional_versions: `{'duckdb': '1.5.4', 'polars': '1.42.1', 'pyarrow': '25.0.0', 'pyspark': None}`
- pandas_version: `2.3.3`
- platform: `macOS-15.5-arm64-arm-64bit`
- processor: `arm`
- python_version: `3.12.13`
- total_ram_bytes: `17179869184`

## Architecture and execution flow

FreshData cases and named pandas component operations run in isolated worker processes. Timing samples and the PeakRSS/tracemalloc measurement are collected separately.

## Reproduction commands

- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-1a2bhs8o/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-1a2bhs8o/result.json shallow_copy`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-1cpbsxvm/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-1cpbsxvm/result.json null_counts`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-2yve61ar/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-2yve61ar/result.json duplicates`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-3t2jaloi/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-3t2jaloi/result.json null_counts`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-3to9120_/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-3to9120_/result.json numeric_median_fill`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-517rp2_i/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-517rp2_i/result.json duplicates`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-5n5kleje/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-5n5kleje/result.json shallow_copy`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-7fgowvwz/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-7fgowvwz/result.json null_counts`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-7w9lzkx2/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-7w9lzkx2/result.json numeric_median_fill`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-8_r4t4do/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-8_r4t4do/result.json numeric_median_fill`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-970pn38n/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-970pn38n/result.json shallow_copy`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-akjg6pa2/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-akjg6pa2/result.json shallow_copy`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-atwivd68/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-atwivd68/result.json duplicates`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-ayxya50_/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-ayxya50_/result.json null_counts`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-dndszzwe/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-dndszzwe/result.json duplicates`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-e39yu2z0/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-e39yu2z0/result.json null_counts`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-e_xkbaob/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-e_xkbaob/result.json numeric_median_fill`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-fw9xjqui/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-fw9xjqui/result.json null_counts`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-g3nkw1m6/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-g3nkw1m6/result.json numeric_median_fill`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-gkg7v_u6/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-gkg7v_u6/result.json duplicates`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-hqxwyk63/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-hqxwyk63/result.json numeric_median_fill`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-hxuyzbdy/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-hxuyzbdy/result.json numeric_median_fill`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-hyt47z_w/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-hyt47z_w/result.json duplicates`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-i35q0qkn/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-i35q0qkn/result.json duplicates`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-i66bg7ug/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-i66bg7ug/result.json null_counts`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-j31qg_8b/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-j31qg_8b/result.json numeric_median_fill`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-jcyb2ewy/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-jcyb2ewy/result.json shallow_copy`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-jml6nw8z/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-jml6nw8z/result.json duplicates`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-kqx67jyy/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-kqx67jyy/result.json duplicates`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-myqv3fk8/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-myqv3fk8/result.json numeric_median_fill`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-nz0pqreo/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-nz0pqreo/result.json null_counts`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-o6r_idke/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-o6r_idke/result.json numeric_median_fill`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-ogfxbv51/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-ogfxbv51/result.json duplicates`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-pe41zuia/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-pe41zuia/result.json shallow_copy`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-r2pbpey9/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-r2pbpey9/result.json null_counts`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-sm2p6tyz/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-sm2p6tyz/result.json null_counts`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-sm2qezvs/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-sm2qezvs/result.json shallow_copy`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-spnaa01k/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-spnaa01k/result.json shallow_copy`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-sv6h_ar5/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-sv6h_ar5/result.json duplicates`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-th5lkqzb/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-th5lkqzb/result.json duplicates`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-tmuuhh49/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-tmuuhh49/result.json shallow_copy`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-u0matknq/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-u0matknq/result.json null_counts`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-vusdqa6w/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-vusdqa6w/result.json numeric_median_fill`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-vuurrf4s/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-vuurrf4s/result.json null_counts`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-wthn21sp/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-wthn21sp/result.json shallow_copy`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-xu2f98r1/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-xu2f98r1/result.json shallow_copy`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-ytg1ke0e/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-ytg1ke0e/result.json shallow_copy`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.baselines import baseline_worker_main; baseline_worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3], __import__('"'"'sys'"'"').argv[4])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-zdipbdz9/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-pandas-baseline-zdipbdz9/result.json numeric_median_fill`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-0_oplliu/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-0_oplliu/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-0o7maroj/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-0o7maroj/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-0wcjge2n/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-0wcjge2n/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-0y608fx9/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-0y608fx9/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-2t3k0qat/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-2t3k0qat/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-35wc2941/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-35wc2941/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-3uzf4yau/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-3uzf4yau/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-3wt9l572/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-3wt9l572/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-45k6rlhy/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-45k6rlhy/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-48bqvd6_/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-48bqvd6_/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-505l5i2k/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-505l5i2k/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-5b5z35kz/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-5b5z35kz/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-5z51_7m7/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-5z51_7m7/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-6f1913hf/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-6f1913hf/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-6h0tex80/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-6h0tex80/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-6m37255r/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-6m37255r/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-6va8ihwu/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-6va8ihwu/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-6vsbnir1/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-6vsbnir1/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-6x8nokur/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-6x8nokur/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-73qnu4tr/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-73qnu4tr/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-7a2ovh8q/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-7a2ovh8q/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-7ijnb2t2/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-7ijnb2t2/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-7kbxkur4/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-7kbxkur4/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-7t8hq6r0/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-7t8hq6r0/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-7vsj5k8x/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-7vsj5k8x/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-7wohf2mk/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-7wohf2mk/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-8z80spay/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-8z80spay/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-94i3izmm/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-94i3izmm/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-9eov5p_h/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-9eov5p_h/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-9nhcsu3f/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-9nhcsu3f/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-9nq36ybo/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-9nq36ybo/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-_2qwkrp5/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-_2qwkrp5/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-_62x0b18/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-_62x0b18/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-_l79249g/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-_l79249g/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-_w7r0q85/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-_w7r0q85/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-a77oq7oj/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-a77oq7oj/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-acfhxjob/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-acfhxjob/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-bfpv5cj2/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-bfpv5cj2/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-bguwaenp/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-bguwaenp/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-bhzt0s5x/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-bhzt0s5x/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-bk_t2v0m/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-bk_t2v0m/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-bkaat7mk/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-bkaat7mk/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-blvv_jiv/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-blvv_jiv/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-bn1sfum9/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-bn1sfum9/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-bvxcn7xl/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-bvxcn7xl/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-cpv66lch/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-cpv66lch/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-cyfb1k82/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-cyfb1k82/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-d7saj_vi/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-d7saj_vi/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-df48de9a/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-df48de9a/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-do203r6b/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-do203r6b/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-dq31_bs5/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-dq31_bs5/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-eazaesl3/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-eazaesl3/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-ed961_m8/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-ed961_m8/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-eudiegfa/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-eudiegfa/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-f264f2_9/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-f264f2_9/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-f2ooeoz0/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-f2ooeoz0/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-fj5psnoh/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-fj5psnoh/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-fro3mtal/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-fro3mtal/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-fxn1gzrg/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-fxn1gzrg/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-g4j166wv/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-g4j166wv/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-g9qfvced/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-g9qfvced/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-g_nftmdn/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-g_nftmdn/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-gc406ib9/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-gc406ib9/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-gxeu66kc/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-gxeu66kc/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-gz_af_1p/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-gz_af_1p/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-h0g7u45k/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-h0g7u45k/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-h_rnavez/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-h_rnavez/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-hlfudhnw/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-hlfudhnw/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-i04nrmrp/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-i04nrmrp/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-i83j742j/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-i83j742j/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-i8zs7mdh/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-i8zs7mdh/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-izm1t_jk/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-izm1t_jk/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-j_rz02un/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-j_rz02un/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-jaqic_qn/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-jaqic_qn/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-jsn72n1l/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-jsn72n1l/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-ksnjrpyp/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-ksnjrpyp/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-kyndmnei/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-kyndmnei/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-l8w0yh4p/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-l8w0yh4p/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-lq544524/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-lq544524/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-lx898arq/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-lx898arq/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-m0sb5i34/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-m0sb5i34/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-m305bi9o/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-m305bi9o/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-mamv5ri_/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-mamv5ri_/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-mk0c_vdo/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-mk0c_vdo/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-mov5obdy/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-mov5obdy/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-ng_asva7/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-ng_asva7/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-nsu1dvvk/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-nsu1dvvk/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-o26qohu4/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-o26qohu4/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-o7uyipn7/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-o7uyipn7/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-obnnpt74/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-obnnpt74/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-oc0p3e55/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-oc0p3e55/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-ofl_se46/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-ofl_se46/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-ofttul9h/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-ofttul9h/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-olcq_0u3/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-olcq_0u3/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-p3db69s_/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-p3db69s_/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-pergwrsg/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-pergwrsg/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-q5jo5ykm/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-q5jo5ykm/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-q6_eotbv/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-q6_eotbv/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-qldc62vn/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-qldc62vn/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-qpv00l7x/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-qpv00l7x/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-quma73mv/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-quma73mv/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-qwq4ttn2/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-qwq4ttn2/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-r1t8ppnu/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-r1t8ppnu/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-r75lh8th/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-r75lh8th/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-rqpltcr2/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-rqpltcr2/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-ru43bw23/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-ru43bw23/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-s12d6qgk/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-s12d6qgk/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-s4rpqe6x/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-s4rpqe6x/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-se9sgn4u/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-se9sgn4u/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-t9v9c099/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-t9v9c099/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-tw8lly9j/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-tw8lly9j/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-u32hxje4/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-u32hxje4/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-uby8hrir/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-uby8hrir/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-uv5wrp3w/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-uv5wrp3w/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-vrmo8_xr/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-vrmo8_xr/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-w5oxtzfm/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-w5oxtzfm/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-w61z76il/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-w61z76il/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-wbkq_rwi/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-wbkq_rwi/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-woqbb2i9/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-woqbb2i9/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-wvyexwm2/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-wvyexwm2/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-wyy78nbb/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-wyy78nbb/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-xhlz9fkb/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-xhlz9fkb/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-xt8z79vb/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-xt8z79vb/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-y1kajzsz/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-y1kajzsz/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-y7aivnpb/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-y7aivnpb/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-y9l2tty4/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-y9l2tty4/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-yybzjsbu/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-yybzjsbu/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-z32pfetj/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-z32pfetj/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-z7x9dr_r/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-z7x9dr_r/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-zqojqylx/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-zqojqylx/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-zsncy0_k/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-zsncy0_k/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -c 'from benchmarks.performance.worker import worker_main; worker_main(__import__('"'"'sys'"'"').argv[1], __import__('"'"'sys'"'"').argv[2], __import__('"'"'sys'"'"').argv[3])' /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-zxw4_spm/case.json /var/folders/5n/6lzstmkn3g906qtsthx5l4b40000gn/T/freshdata-benchmark-zxw4_spm/result.json`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -m benchmarks.performance profile --rows 10000 --widths wide --configs default --report-modes true --output benchmarks/results/performance/baseline`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -m benchmarks.performance profile --rows 100000 --widths medium --configs aggressive --report-modes true --output benchmarks/results/performance/baseline`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -m benchmarks.performance profile --rows 100000 --widths medium --configs default --report-modes true --output benchmarks/results/performance/baseline`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -m benchmarks.performance profile --rows 100000 --widths medium --configs semantic --report-modes true --output benchmarks/results/performance/baseline`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -m benchmarks.performance profile --rows 1000000 --widths narrow --configs default --report-modes true --output benchmarks/results/performance/baseline`
- `/Users/wilson/freshdata-qa/.venv-qa/bin/python -m benchmarks.performance profile --rows 500000 --widths medium --configs default --report-modes false --output benchmarks/results/performance/baseline`

## Baseline benchmark table

| Rows | Width | Config / operation | Report | Backend | Format | Median s | Min s | Max s | CV | Rows/s | Peak RSS/input | Peak Python/input | Comparison |
| ---: | :--- | :--- | :---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 10000 | medium | conservative | false | pandas | pandas | 0.306 | 0.306 | 0.307 | 0.001 | 32659.8 | 1.686 | 1.429 | — |
| 10000 | medium | conservative | true | pandas | pandas | 0.301 | 0.301 | 0.301 | 0.001 | 33197.3 | 2.461 | 1.429 | — |
| 10000 | medium | default | false | pandas | pandas | 0.428 | 0.428 | 0.432 | 0.004 | 23359.8 | 0.810 | 1.429 | — |
| 10000 | medium | default | true | pandas | pandas | 0.427 | 0.426 | 0.430 | 0.004 | 23438.4 | 1.040 | 1.430 | — |
| 10000 | medium | explicit | false | pandas | pandas | 0.331 | 0.328 | 0.342 | 0.020 | 30231.9 | 0.365 | 1.429 | — |
| 10000 | medium | explicit | true | pandas | pandas | 0.333 | 0.331 | 0.348 | 0.023 | 30022.1 | 0.808 | 1.430 | — |
| 10000 | medium | duplicates | false | pandas-component-baseline | pandas | 0.019 | 0.019 | 0.019 | 0.014 | 530246.1 | 0.190 | 0.659 | — |
| 10000 | medium | null_counts | false | pandas-component-baseline | pandas | 0.004 | 0.004 | 0.004 | 0.025 | 2651230.1 | 0.051 | 0.137 | — |
| 10000 | medium | shallow_copy | false | pandas-component-baseline | pandas | 0.000 | 0.000 | 0.000 | 0.197 | 132522296.9 | 0.004 | 0.081 | — |
| 10000 | medium | representation_off | false | pandas | pandas | 0.046 | 0.044 | 0.058 | 0.124 | 215696.2 | 0.051 | 0.149 | — |
| 10000 | medium | representation_off | true | pandas | pandas | 0.045 | 0.044 | 0.047 | 0.021 | 220997.5 | 0.053 | 0.149 | — |
| 10000 | medium | statistical_off | false | pandas | pandas | 0.311 | 0.307 | 0.328 | 0.027 | 32112.4 | 0.332 | 1.429 | — |
| 10000 | medium | statistical_off | true | pandas | pandas | 0.305 | 0.302 | 0.307 | 0.007 | 32790.8 | 1.773 | 1.429 | — |
| 10000 | narrow | conservative | false | pandas | pandas | 0.071 | 0.071 | 0.071 | 0.004 | 140676.2 | 2.342 | 2.233 | — |
| 10000 | narrow | conservative | true | pandas | pandas | 0.071 | 0.071 | 0.072 | 0.005 | 140136.2 | 1.650 | 2.233 | — |
| 10000 | narrow | default | false | pandas | pandas | 0.102 | 0.102 | 0.103 | 0.002 | 97739.7 | 2.933 | 2.233 | — |
| 10000 | narrow | default | true | pandas | pandas | 0.102 | 0.101 | 0.103 | 0.007 | 97700.7 | 1.802 | 2.233 | — |
| 10000 | narrow | explicit | false | pandas | pandas | 0.077 | 0.077 | 0.079 | 0.013 | 129628.8 | 1.008 | 2.233 | — |
| 10000 | narrow | explicit | true | pandas | pandas | 0.077 | 0.077 | 0.077 | 0.001 | 129772.8 | 2.882 | 2.233 | — |
| 10000 | narrow | duplicates | false | pandas-component-baseline | pandas | 0.005 | 0.004 | 0.005 | 0.090 | 2170295.7 | 0.061 | 1.434 | — |
| 10000 | narrow | null_counts | false | pandas-component-baseline | pandas | 0.001 | 0.001 | 0.001 | 0.056 | 10384216.0 | 0.051 | 0.458 | — |
| 10000 | narrow | shallow_copy | false | pandas-component-baseline | pandas | 0.000 | 0.000 | 0.000 | 0.085 | 197367122.2 | 0.031 | 0.375 | — |
| 10000 | narrow | representation_off | false | pandas | pandas | 0.010 | 0.009 | 0.010 | 0.014 | 1050020.3 | 0.132 | 0.478 | — |
| 10000 | narrow | representation_off | true | pandas | pandas | 0.010 | 0.010 | 0.010 | 0.011 | 1028127.8 | 0.112 | 0.478 | — |
| 10000 | narrow | statistical_off | false | pandas | pandas | 0.071 | 0.071 | 0.072 | 0.003 | 139984.1 | 1.955 | 2.233 | — |
| 10000 | narrow | statistical_off | true | pandas | pandas | 0.077 | 0.077 | 0.079 | 0.011 | 129739.8 | 0.876 | 2.233 | — |
| 10000 | wide | conservative | false | pandas | pandas | 1.244 | 1.242 | 1.254 | 0.004 | 8037.4 | 1.771 | 1.222 | — |
| 10000 | wide | conservative | true | pandas | pandas | 1.252 | 1.247 | 1.268 | 0.008 | 7985.3 | 0.101 | 1.223 | — |
| 10000 | wide | default | false | pandas | pandas | 1.855 | 1.845 | 1.867 | 0.005 | 5391.5 | 0.061 | 1.223 | — |
| 10000 | wide | default | true | pandas | pandas | 1.865 | 1.843 | 1.882 | 0.008 | 5362.0 | 1.158 | 1.223 | — |
| 10000 | wide | explicit | false | pandas | pandas | 1.319 | 1.314 | 1.322 | 0.002 | 7582.1 | 1.272 | 1.222 | — |
| 10000 | wide | explicit | true | pandas | pandas | 1.329 | 1.323 | 1.338 | 0.005 | 7525.3 | 0.577 | 1.223 | — |
| 10000 | wide | duplicates | false | pandas-component-baseline | pandas | 0.082 | 0.079 | 0.141 | 0.331 | 122453.0 | 0.048 | 0.493 | — |
| 10000 | wide | null_counts | false | pandas-component-baseline | pandas | 0.015 | 0.015 | 0.015 | 0.013 | 665581.4 | 0.030 | 0.069 | — |
| 10000 | wide | shallow_copy | false | pandas-component-baseline | pandas | 0.000 | 0.000 | 0.000 | 0.213 | 62794348.5 | 0.002 | 0.020 | — |
| 10000 | wide | representation_off | false | pandas | pandas | 0.181 | 0.180 | 0.183 | 0.005 | 55178.7 | 0.058 | 0.080 | — |
| 10000 | wide | representation_off | true | pandas | pandas | 0.182 | 0.181 | 0.182 | 0.003 | 54988.2 | 0.057 | 0.080 | — |
| 10000 | wide | statistical_off | false | pandas | pandas | 1.241 | 1.239 | 1.247 | 0.003 | 8058.1 | 0.209 | 1.222 | — |
| 10000 | wide | statistical_off | true | pandas | pandas | 1.253 | 1.247 | 1.296 | 0.016 | 7983.3 | 0.090 | 1.223 | — |
| 100000 | medium | default | false | pandas | pandas | 0.703 | 0.702 | 0.734 | 0.020 | 142223.8 | 4.615 | 13.956 | — |
| 100000 | medium | default | true | pandas | pandas | 0.692 | 0.683 | 0.713 | 0.016 | 144449.4 | 10.199 | 13.955 | — |
| 100000 | medium | default | false | pandas | pandas | 0.548 | 0.547 | 0.564 | 0.013 | 182445.3 | 0.084 | 2.068 | — |
| 100000 | medium | default | true | pandas | pandas | 0.550 | 0.550 | 0.567 | 0.014 | 181838.5 | 1.144 | 2.068 | — |
| 100000 | medium | default | false | pandas | pandas | 7.134 | 7.076 | 7.384 | 0.018 | 14017.2 | 0.062 | 0.402 | — |
| 100000 | medium | default | true | pandas | pandas | 6.888 | 6.834 | 6.947 | 0.007 | 14518.7 | 0.051 | 0.402 | — |
| 100000 | medium | aggressive | true | pandas | pandas | 2.814 | 2.810 | 2.844 | 0.005 | 35534.9 | 0.297 | 1.324 | — |
| 100000 | medium | conservative | false | pandas | pandas | 1.906 | 1.881 | 2.228 | 0.077 | 52455.9 | 0.402 | 1.332 | — |
| 100000 | medium | conservative | true | pandas | pandas | 1.912 | 1.893 | 2.092 | 0.044 | 52306.7 | 0.209 | 1.332 | — |
| 100000 | medium | default | false | pandas | pandas | 2.766 | 2.740 | 2.782 | 0.006 | 36151.7 | 0.436 | 1.332 | — |
| 100000 | medium | default | true | pandas | pandas | 3.707 | 2.931 | 4.235 | 0.135 | 26976.0 | 0.408 | 1.332 | — |
| 100000 | medium | explicit | false | pandas | pandas | 2.000 | 1.978 | 2.018 | 0.008 | 50001.7 | 0.353 | 1.332 | — |
| 100000 | medium | explicit | true | pandas | pandas | 2.180 | 2.000 | 3.327 | 0.249 | 45880.6 | 0.201 | 1.332 | — |
| 100000 | medium | duplicates | false | pandas-component-baseline | pandas | 0.246 | 0.223 | 0.269 | 0.075 | 405835.5 | 0.023 | 0.555 | — |
| 100000 | medium | null_counts | false | pandas-component-baseline | pandas | 0.036 | 0.036 | 0.036 | 0.004 | 2806275.8 | 0.001 | 0.052 | — |
| 100000 | medium | shallow_copy | false | pandas-component-baseline | pandas | 0.000 | 0.000 | 0.000 | 0.613 | 656995690.0 | 0.000 | 0.008 | — |
| 100000 | medium | representation_off | false | pandas | pandas | 0.405 | 0.400 | 0.409 | 0.010 | 246643.4 | 0.043 | 0.056 | — |
| 100000 | medium | representation_off | true | pandas | pandas | 0.412 | 0.411 | 0.415 | 0.003 | 242509.9 | 0.033 | 0.056 | — |
| 100000 | medium | semantic | true | pandas | pandas | 4.147 | 4.133 | 4.198 | 0.008 | 24112.4 | 0.205 | 1.332 | — |
| 100000 | medium | statistical_off | false | pandas | pandas | 1.908 | 1.890 | 2.272 | 0.085 | 52411.9 | 0.257 | 1.332 | — |
| 100000 | medium | statistical_off | true | pandas | pandas | 1.931 | 1.881 | 2.256 | 0.080 | 51781.5 | 0.200 | 1.332 | — |
| 100000 | medium | default | false | pandas | pandas | 1.481 | 1.444 | 1.584 | 0.035 | 67510.6 | 1.203 | 3.055 | — |
| 100000 | medium | default | true | pandas | pandas | 1.561 | 1.508 | 1.632 | 0.035 | 64059.3 | 0.101 | 3.055 | — |
| 100000 | medium | default | false | pandas | pandas | 1.112 | 1.105 | 1.122 | 0.007 | 89925.3 | 0.152 | 3.183 | — |
| 100000 | medium | default | true | pandas | pandas | 1.115 | 1.108 | 1.124 | 0.006 | 89665.9 | 0.031 | 3.183 | — |
| 100000 | medium | default | false | pandas | pandas | 5.241 | 5.202 | 5.265 | 0.005 | 19081.8 | 0.063 | 0.398 | — |
| 100000 | medium | default | true | pandas | pandas | 6.392 | 6.249 | 6.513 | 0.019 | 15645.7 | 0.056 | 0.398 | — |
| 100000 | narrow | conservative | false | pandas | pandas | 0.467 | 0.467 | 0.470 | 0.003 | 213928.2 | 2.007 | 1.868 | — |
| 100000 | narrow | conservative | true | pandas | pandas | 0.477 | 0.471 | 0.481 | 0.009 | 209503.1 | 1.202 | 1.868 | — |
| 100000 | narrow | default | false | pandas | pandas | 0.671 | 0.666 | 0.675 | 0.005 | 148961.6 | 1.297 | 1.868 | — |
| 100000 | narrow | default | true | pandas | pandas | 0.670 | 0.665 | 0.670 | 0.004 | 149284.4 | 1.709 | 1.868 | — |
| 100000 | narrow | explicit | false | pandas | pandas | 0.505 | 0.501 | 0.507 | 0.005 | 197974.5 | 1.871 | 1.868 | — |
| 100000 | narrow | explicit | true | pandas | pandas | 0.501 | 0.494 | 0.502 | 0.008 | 199726.3 | 2.063 | 1.868 | — |
| 100000 | narrow | duplicates | false | pandas-component-baseline | pandas | 0.061 | 0.052 | 0.075 | 0.132 | 1646529.8 | 0.006 | 0.973 | — |
| 100000 | narrow | null_counts | false | pandas-component-baseline | pandas | 0.009 | 0.007 | 0.009 | 0.088 | 11236428.5 | 0.005 | 0.090 | — |
| 100000 | narrow | numeric_median_fill | false | pandas-component-baseline | pandas | 0.016 | 0.014 | 0.017 | 0.086 | 6329497.7 | 0.262 | 0.581 | — |
| 100000 | narrow | shallow_copy | false | pandas-component-baseline | pandas | 0.000 | 0.000 | 0.000 | 0.821 | 1584158415.9 | 0.003 | 0.037 | — |
| 100000 | narrow | representation_off | false | pandas | pandas | 0.083 | 0.083 | 0.088 | 0.026 | 1204377.9 | 0.079 | 0.105 | — |
| 100000 | narrow | representation_off | true | pandas | pandas | 0.083 | 0.083 | 0.083 | 0.003 | 1202504.2 | 0.014 | 0.105 | — |
| 100000 | narrow | statistical_off | false | pandas | pandas | 0.473 | 0.470 | 0.473 | 0.003 | 211589.5 | 0.963 | 1.868 | — |
| 100000 | narrow | statistical_off | true | pandas | pandas | 0.469 | 0.467 | 0.473 | 0.005 | 213185.3 | 0.807 | 1.868 | — |
| 100000 | wide | conservative | false | pandas | pandas | 8.714 | 8.670 | 8.860 | 0.009 | 11476.1 | 0.278 | 1.182 | — |
| 100000 | wide | conservative | true | pandas | pandas | 7.734 | 7.666 | 7.745 | 0.004 | 12929.9 | 0.056 | 1.182 | — |
| 100000 | wide | default | false | pandas | pandas | 13.121 | 13.022 | 13.372 | 0.010 | 7621.3 | 0.054 | 1.182 | — |
| 100000 | wide | default | true | pandas | pandas | 12.393 | 11.614 | 13.080 | 0.050 | 8069.3 | 0.270 | 1.182 | — |
| 100000 | wide | explicit | false | pandas | pandas | 8.066 | 8.032 | 8.125 | 0.005 | 12398.2 | 0.492 | 1.182 | — |
| 100000 | wide | explicit | true | pandas | pandas | 7.975 | 7.944 | 8.028 | 0.005 | 12539.2 | 0.415 | 1.182 | — |
| 100000 | wide | duplicates | false | pandas-component-baseline | pandas | 0.993 | 0.980 | 1.126 | 0.063 | 100671.2 | 0.003 | 0.465 | — |
| 100000 | wide | null_counts | false | pandas-component-baseline | pandas | 0.171 | 0.164 | 0.203 | 0.092 | 586268.0 | 0.018 | 0.044 | — |
| 100000 | wide | shallow_copy | false | pandas-component-baseline | pandas | 0.000 | 0.000 | 0.000 | 0.110 | 523560209.5 | 0.000 | 0.002 | — |
| 100000 | wide | representation_off | false | pandas | pandas | 1.687 | 1.682 | 1.697 | 0.004 | 59261.1 | 0.002 | 0.045 | — |
| 100000 | wide | representation_off | true | pandas | pandas | 1.710 | 1.699 | 1.726 | 0.006 | 58469.8 | 0.003 | 0.045 | — |
| 100000 | wide | statistical_off | false | pandas | pandas | 7.653 | 7.600 | 7.695 | 0.006 | 13066.8 | 0.058 | 1.182 | — |
| 100000 | wide | statistical_off | true | pandas | pandas | 7.658 | 7.646 | 7.939 | 0.020 | 13057.9 | 0.142 | 1.182 | — |
| 500000 | medium | conservative | false | pandas | pandas | 9.956 | 9.922 | 10.239 | 0.013 | 50222.8 | 0.167 | 1.316 | — |
| 500000 | medium | conservative | true | pandas | pandas | 9.827 | 9.785 | 9.937 | 0.006 | 50879.3 | 0.169 | 1.316 | — |
| 500000 | medium | default | false | pandas | pandas | 20.539 | 19.755 | 20.979 | 0.023 | 24343.8 | 0.158 | 1.316 | — |
| 500000 | medium | default | true | pandas | pandas | 19.412 | 18.143 | 23.753 | 0.131 | 25757.6 | 0.203 | 1.316 | — |
| 500000 | medium | explicit | false | pandas | pandas | 8.363 | 8.292 | 10.263 | 0.103 | 59784.2 | 0.163 | 1.316 | — |
| 500000 | medium | explicit | true | pandas | pandas | 8.495 | 8.386 | 8.552 | 0.008 | 58856.5 | 0.163 | 1.316 | — |
| 500000 | medium | duplicates | false | pandas-component-baseline | pandas | 2.028 | 1.402 | 2.856 | 0.333 | 246536.8 | 0.000 | 0.559 | — |
| 500000 | medium | null_counts | false | pandas-component-baseline | pandas | 0.179 | 0.165 | 0.306 | 0.338 | 2786726.9 | 0.000 | 0.045 | — |
| 500000 | medium | shallow_copy | false | pandas-component-baseline | pandas | 0.000 | 0.000 | 0.000 | 0.152 | 5695279749.2 | 0.000 | 0.002 | — |
| 500000 | medium | representation_off | false | pandas | pandas | 0.834 | 0.833 | 0.863 | 0.016 | 599344.3 | 0.001 | 0.048 | — |
| 500000 | medium | representation_off | true | pandas | pandas | 0.828 | 0.827 | 0.830 | 0.002 | 604101.6 | 0.001 | 0.048 | — |
| 500000 | medium | statistical_off | false | pandas | pandas | 9.960 | 9.821 | 10.470 | 0.025 | 50201.2 | 0.239 | 1.316 | — |
| 500000 | medium | statistical_off | true | pandas | pandas | 10.277 | 10.094 | 10.650 | 0.025 | 48654.7 | 0.190 | 1.316 | — |
| 500000 | narrow | conservative | false | pandas | pandas | 2.057 | 2.056 | 2.066 | 0.002 | 243017.2 | 0.886 | 1.827 | — |
| 500000 | narrow | conservative | true | pandas | pandas | 2.057 | 2.051 | 2.062 | 0.002 | 243071.6 | 0.862 | 1.827 | — |
| 500000 | narrow | default | false | pandas | pandas | 3.186 | 3.149 | 3.214 | 0.008 | 156918.3 | 0.953 | 1.827 | — |
| 500000 | narrow | default | true | pandas | pandas | 3.660 | 3.630 | 3.795 | 0.018 | 136609.5 | 1.025 | 1.827 | — |
| 500000 | narrow | explicit | false | pandas | pandas | 2.167 | 2.153 | 2.180 | 0.005 | 230747.3 | 1.921 | 1.827 | — |
| 500000 | narrow | explicit | true | pandas | pandas | 2.171 | 2.162 | 2.230 | 0.013 | 230312.5 | 1.281 | 1.827 | — |
| 500000 | narrow | duplicates | false | pandas-component-baseline | pandas | 0.438 | 0.314 | 0.560 | 0.259 | 1141796.9 | 0.189 | 1.004 | — |
| 500000 | narrow | null_counts | false | pandas-component-baseline | pandas | 0.034 | 0.034 | 0.035 | 0.010 | 14522429.9 | 0.000 | 0.057 | — |
| 500000 | narrow | numeric_median_fill | false | pandas-component-baseline | pandas | 0.061 | 0.060 | 0.064 | 0.030 | 8247394.4 | 0.007 | 0.548 | — |
| 500000 | narrow | shallow_copy | false | pandas-component-baseline | pandas | 0.000 | 0.000 | 0.000 | 0.071 | 7159836184.3 | 0.000 | 0.007 | — |
| 500000 | narrow | representation_off | false | pandas | pandas | 0.231 | 0.231 | 0.231 | 0.000 | 2163479.7 | 0.056 | 0.075 | — |
| 500000 | narrow | representation_off | true | pandas | pandas | 0.163 | 0.163 | 0.164 | 0.002 | 3062243.9 | 0.108 | 0.075 | — |
| 500000 | narrow | statistical_off | false | pandas | pandas | 2.057 | 2.030 | 2.065 | 0.007 | 243067.0 | 0.873 | 1.827 | — |
| 500000 | narrow | statistical_off | true | pandas | pandas | 2.049 | 2.039 | 2.134 | 0.019 | 244003.2 | 0.971 | 1.827 | — |
| 500000 | wide | conservative | false | pandas | pandas | 30.908 | 30.745 | 31.115 | 0.004 | 16177.3 | 0.310 | 1.172 | — |
| 500000 | wide | conservative | true | pandas | pandas | 30.906 | 30.861 | 30.935 | 0.001 | 16178.2 | 0.369 | 1.172 | — |
| 500000 | wide | default | false | pandas | pandas | 52.681 | 52.298 | 54.534 | 0.018 | 9491.1 | 0.318 | 1.172 | — |
| 500000 | wide | default | true | pandas | pandas | 69.694 | 54.972 | 70.916 | 0.096 | 7174.2 | 0.352 | 1.172 | — |
| 500000 | wide | explicit | false | pandas | pandas | 32.221 | 32.133 | 32.415 | 0.004 | 15518.0 | 0.542 | 1.172 | — |
| 500000 | wide | explicit | true | pandas | pandas | 33.529 | 32.020 | 36.898 | 0.055 | 14912.3 | 0.173 | 1.172 | — |
| 500000 | wide | duplicates | false | pandas-component-baseline | pandas | 4.743 | 4.591 | 5.050 | 0.037 | 105418.0 | 0.000 | 0.464 | — |
| 500000 | wide | null_counts | false | pandas-component-baseline | pandas | 0.678 | 0.655 | 0.718 | 0.038 | 737053.2 | 0.001 | 0.042 | — |
| 500000 | wide | shallow_copy | false | pandas-component-baseline | pandas | 0.000 | 0.000 | 0.000 | 0.097 | 2761149521.5 | 0.000 | 0.000 | — |
| 500000 | wide | representation_off | false | pandas | pandas | 3.499 | 3.489 | 3.505 | 0.002 | 142887.5 | 0.001 | 0.043 | — |
| 500000 | wide | representation_off | true | pandas | pandas | 3.542 | 3.524 | 3.672 | 0.018 | 141179.7 | 0.000 | 0.043 | — |
| 500000 | wide | statistical_off | false | pandas | pandas | 30.767 | 30.633 | 30.926 | 0.004 | 16251.2 | 0.341 | 1.172 | — |
| 500000 | wide | statistical_off | true | pandas | pandas | 30.559 | 30.448 | 30.594 | 0.002 | 16361.7 | 0.371 | 1.172 | — |
| 1000000 | medium | conservative | false | pandas | pandas | 15.863 | 15.793 | 15.908 | 0.003 | 63041.3 | 0.192 | 1.314 | — |
| 1000000 | medium | conservative | true | pandas | pandas | 15.932 | 15.816 | 16.166 | 0.009 | 62765.2 | 0.087 | 1.314 | — |
| 1000000 | medium | default | false | pandas | pandas | 27.221 | 26.601 | 27.344 | 0.011 | 36735.9 | 0.283 | 1.314 | — |
| 1000000 | medium | default | true | pandas | pandas | 26.805 | 26.580 | 26.889 | 0.005 | 37307.0 | 0.344 | 1.314 | — |
| 1000000 | medium | explicit | false | pandas | pandas | 16.990 | 16.890 | 17.078 | 0.005 | 58859.3 | 0.155 | 1.314 | — |
| 1000000 | medium | explicit | true | pandas | pandas | 17.032 | 16.976 | 17.573 | 0.015 | 58712.8 | 0.197 | 1.314 | — |
| 1000000 | medium | duplicates | false | pandas-component-baseline | pandas | 2.671 | 2.417 | 2.840 | 0.064 | 374394.7 | 0.163 | 0.569 | — |
| 1000000 | medium | null_counts | false | pandas-component-baseline | pandas | 0.370 | 0.356 | 0.477 | 0.145 | 2700703.0 | 0.000 | 0.044 | — |
| 1000000 | medium | numeric_median_fill | false | pandas-component-baseline | pandas | 0.432 | 0.425 | 0.434 | 0.008 | 2315979.5 | 0.000 | 0.570 | — |
| 1000000 | medium | shallow_copy | false | pandas-component-baseline | pandas | 0.000 | 0.000 | 0.000 | 0.203 | 13392976714.8 | 0.000 | 0.001 | — |
| 1000000 | medium | representation_off | false | pandas | pandas | 1.587 | 1.584 | 1.637 | 0.014 | 629979.9 | 0.000 | 0.048 | — |
| 1000000 | medium | representation_off | true | pandas | pandas | 1.579 | 1.573 | 1.584 | 0.003 | 633259.2 | 0.000 | 0.048 | — |
| 1000000 | medium | statistical_off | false | pandas | pandas | 15.760 | 15.642 | 15.905 | 0.006 | 63453.4 | 0.120 | 1.314 | — |
| 1000000 | medium | statistical_off | true | pandas | pandas | 15.801 | 15.787 | 15.979 | 0.005 | 63285.8 | 0.377 | 1.314 | — |
| 1000000 | narrow | conservative | false | pandas | pandas | 4.203 | 4.200 | 4.259 | 0.006 | 237950.8 | 1.034 | 1.822 | — |
| 1000000 | narrow | conservative | true | pandas | pandas | 4.287 | 4.273 | 4.303 | 0.003 | 233270.1 | 1.120 | 1.822 | — |
| 1000000 | narrow | default | false | pandas | pandas | 8.903 | 8.241 | 10.087 | 0.094 | 112323.8 | 0.966 | 1.822 | — |
| 1000000 | narrow | default | true | pandas | pandas | 6.910 | 6.882 | 7.062 | 0.012 | 144724.3 | 0.961 | 1.822 | — |
| 1000000 | narrow | explicit | false | pandas | pandas | 4.980 | 4.879 | 5.141 | 0.020 | 200800.4 | 0.994 | 1.822 | — |
| 1000000 | narrow | explicit | true | pandas | pandas | 5.722 | 5.472 | 6.330 | 0.059 | 174772.8 | 0.991 | 1.822 | — |
| 1000000 | narrow | duplicates | false | pandas-component-baseline | pandas | 0.593 | 0.573 | 0.682 | 0.076 | 1686888.2 | 0.000 | 1.051 | — |
| 1000000 | narrow | null_counts | false | pandas-component-baseline | pandas | 0.060 | 0.060 | 0.071 | 0.083 | 16712847.6 | 0.032 | 0.053 | — |
| 1000000 | narrow | numeric_median_fill | false | pandas-component-baseline | pandas | 0.128 | 0.109 | 0.133 | 0.092 | 7831962.6 | 0.000 | 0.544 | — |
| 1000000 | narrow | shallow_copy | false | pandas-component-baseline | pandas | 0.000 | 0.000 | 0.000 | 0.226 | 15635260626.3 | 0.000 | 0.004 | — |
| 1000000 | narrow | representation_off | false | pandas | pandas | 0.316 | 0.315 | 0.318 | 0.003 | 3162296.2 | 0.001 | 0.071 | — |
| 1000000 | narrow | representation_off | true | pandas | pandas | 0.313 | 0.313 | 0.315 | 0.003 | 3189867.3 | 0.000 | 0.071 | — |
| 1000000 | narrow | statistical_off | false | pandas | pandas | 4.274 | 4.251 | 4.300 | 0.004 | 233993.7 | 0.980 | 1.822 | — |
| 1000000 | narrow | statistical_off | true | pandas | pandas | 4.331 | 4.292 | 4.646 | 0.039 | 230905.6 | 0.978 | 1.822 | — |
| 1000000 | wide | conservative | false | pandas | pandas | 67.080 | 65.213 | 67.988 | 0.017 | 14907.5 | 0.647 | 1.171 | — |
| 1000000 | wide | conservative | true | pandas | pandas | 64.627 | 64.291 | 65.270 | 0.006 | 15473.3 | 0.475 | 1.171 | — |
| 1000000 | wide | default | false | pandas | pandas | 112.110 | 109.789 | 113.332 | 0.013 | 8919.8 | 0.449 | 1.171 | — |
| 1000000 | wide | default | true | pandas | pandas | 110.060 | 109.935 | 110.329 | 0.001 | 9085.9 | 0.326 | 1.171 | — |
| 1000000 | wide | explicit | false | pandas | pandas | 65.768 | 65.534 | 66.738 | 0.008 | 15204.9 | 0.651 | 1.171 | — |
| 1000000 | wide | explicit | true | pandas | pandas | 65.115 | 64.894 | 66.087 | 0.007 | 15357.4 | 0.546 | 1.171 | — |
| 1000000 | wide | duplicates | false | pandas-component-baseline | pandas | 13.899 | 13.198 | 17.881 | 0.135 | 71945.6 | 0.124 | 0.466 | — |
| 1000000 | wide | null_counts | false | pandas-component-baseline | pandas | 1.264 | 1.246 | 1.266 | 0.007 | 791195.9 | 0.013 | 0.041 | — |
| 1000000 | wide | numeric_median_fill | false | pandas-component-baseline | pandas | 2.921 | 2.759 | 4.580 | 0.258 | 342326.2 | 0.158 | 0.564 | — |
| 1000000 | wide | shallow_copy | false | pandas-component-baseline | pandas | 0.000 | 0.000 | 0.000 | 0.375 | 4426404390.3 | 0.000 | 0.000 | — |
| 1000000 | wide | representation_off | false | pandas | pandas | 6.676 | 6.619 | 6.704 | 0.006 | 149796.8 | 0.001 | 0.042 | — |
| 1000000 | wide | representation_off | true | pandas | pandas | 6.899 | 6.862 | 7.027 | 0.010 | 144950.1 | 0.000 | 0.042 | — |
| 1000000 | wide | statistical_off | false | pandas | pandas | 65.252 | 64.263 | 66.477 | 0.013 | 15325.3 | 0.372 | 1.171 | — |
| 1000000 | wide | statistical_off | true | pandas | pandas | 68.690 | 68.404 | 69.548 | 0.007 | 14558.1 | 0.401 | 1.171 | — |

## Profiling findings

### case_id=2e88248a20ec642b rows=10000 width=wide dataset_type=mixed config=default options={"verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5

- audit_events: 0.0%
- backend_conversion: 0.0%
- context: 0.2%
- correlation: 0.2%
- dtype_repair: 2.5%
- duplicates: 0.1%
- engine_cache: 0.0%
- missing: 0.0%
- outliers: 0.0%
- report_finalization: 0.0%
- role_inference: 0.1%
- semantic_ml: 0.0%
- `/Users/wilson/freshdata-qa/src/freshdata/api.py:138 clean` — self 0.000 s, cumulative 18.448 s, calls 1
- `/Users/wilson/freshdata-qa/src/freshdata/cleaner.py:189 clean` — self 0.000 s, cumulative 18.444 s, calls 1
- `/Users/wilson/freshdata-qa/src/freshdata/cleaner.py:45 run_pipeline` — self 0.002 s, cumulative 18.444 s, calls 1
- `/Users/wilson/freshdata-qa/src/freshdata/steps/dtypes.py:332 fix_dtypes` — self 0.002 s, cumulative 11.889 s, calls 1
- `/Users/wilson/freshdata-qa/src/freshdata/steps/dtypes.py:297 suggest_conversion` — self 0.002 s, cumulative 11.884 s, calls 42
- `/Users/wilson/freshdata-qa/src/freshdata/steps/dtypes.py:133 _try_numeric` — self 0.002 s, cumulative 10.877 s, calls 42
- `/Users/wilson/freshdata-qa/src/freshdata/steps/dtypes.py:124 _to_numeric_or_none` — self 0.000 s, cumulative 8.525 s, calls 42
- `/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/tools/numeric.py:47 to_numeric` — self 8.505 s, cumulative 8.524 s, calls 42
- `/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/strings/accessor.py:132 wrapper` — self 0.001 s, cumulative 2.324 s, calls 210
- `/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/string_.py:431 _str_map` — self 0.050 s, cumulative 2.239 s, calls 210

### case_id=a877f296485e3003 rows=100000 width=medium dataset_type=mixed config=aggressive options={"strategy":"aggressive","verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5

- audit_events: 0.0%
- backend_conversion: 0.0%
- context: 0.0%
- correlation: 0.2%
- dtype_repair: 1.8%
- duplicates: 0.3%
- engine_cache: 0.0%
- missing: 0.0%
- outliers: 0.0%
- report_finalization: 0.0%
- role_inference: 0.1%
- semantic_ml: 0.0%
- `/Users/wilson/freshdata-qa/src/freshdata/api.py:138 clean` — self 0.000 s, cumulative 15.656 s, calls 1
- `/Users/wilson/freshdata-qa/src/freshdata/cleaner.py:189 clean` — self 0.000 s, cumulative 15.655 s, calls 1
- `/Users/wilson/freshdata-qa/src/freshdata/cleaner.py:45 run_pipeline` — self 0.002 s, cumulative 15.655 s, calls 1
- `/Users/wilson/freshdata-qa/src/freshdata/steps/dtypes.py:332 fix_dtypes` — self 0.003 s, cumulative 4.873 s, calls 1
- `/Users/wilson/freshdata-qa/src/freshdata/steps/dtypes.py:297 suggest_conversion` — self 0.003 s, cumulative 4.869 s, calls 10
- `/Users/wilson/freshdata-qa/src/freshdata/steps/duplicates.py:83 drop_duplicate_rows` — self 0.003 s, cumulative 3.694 s, calls 1
- `/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/strings/accessor.py:132 wrapper` — self 0.000 s, cumulative 3.660 s, calls 50
- `/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/string_.py:431 _str_map` — self 0.012 s, cumulative 3.635 s, calls 50
- `/Users/wilson/freshdata-qa/src/freshdata/_util.py:34 memory_bytes` — self 0.000 s, cumulative 3.352 s, calls 2
- `/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/frame.py:3677 memory_usage` — self 0.000 s, cumulative 3.351 s, calls 2

### case_id=eb41d9a6034039f8 rows=100000 width=medium dataset_type=mixed config=default options={"verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5

- audit_events: 0.0%
- backend_conversion: 0.0%
- context: 0.1%
- correlation: 0.2%
- dtype_repair: 1.6%
- duplicates: 0.2%
- engine_cache: 0.0%
- missing: 0.0%
- outliers: 0.0%
- report_finalization: 0.0%
- role_inference: 0.1%
- semantic_ml: 0.0%
- `/Users/wilson/freshdata-qa/src/freshdata/api.py:138 clean` — self 0.000 s, cumulative 16.007 s, calls 1
- `/Users/wilson/freshdata-qa/src/freshdata/cleaner.py:189 clean` — self 0.000 s, cumulative 16.006 s, calls 1
- `/Users/wilson/freshdata-qa/src/freshdata/cleaner.py:45 run_pipeline` — self 0.001 s, cumulative 16.006 s, calls 1
- `/Users/wilson/freshdata-qa/src/freshdata/steps/dtypes.py:332 fix_dtypes` — self 0.003 s, cumulative 4.828 s, calls 1
- `/Users/wilson/freshdata-qa/src/freshdata/steps/dtypes.py:297 suggest_conversion` — self 0.003 s, cumulative 4.825 s, calls 10
- `/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/strings/accessor.py:132 wrapper` — self 0.000 s, cumulative 3.771 s, calls 50
- `/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/string_.py:431 _str_map` — self 0.012 s, cumulative 3.745 s, calls 50
- `/Users/wilson/freshdata-qa/src/freshdata/steps/duplicates.py:83 drop_duplicate_rows` — self 0.003 s, cumulative 3.697 s, calls 1
- `/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/string_.py:486 _str_map_str_or_object` — self 1.147 s, cumulative 3.452 s, calls 40
- `/Users/wilson/freshdata-qa/src/freshdata/_util.py:34 memory_bytes` — self 0.000 s, cumulative 3.319 s, calls 2

### case_id=9ea9cb03dfc114c5 rows=100000 width=medium dataset_type=mixed config=semantic options={"semantic_mode":"assist","verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5

- audit_events: 0.0%
- backend_conversion: 0.0%
- context: 0.1%
- correlation: 0.1%
- dtype_repair: 1.2%
- duplicates: 0.2%
- engine_cache: 0.0%
- missing: 0.0%
- outliers: 0.0%
- report_finalization: 0.0%
- role_inference: 0.1%
- semantic_ml: 13.1%
- `/Users/wilson/freshdata-qa/src/freshdata/api.py:138 clean` — self 0.000 s, cumulative 22.017 s, calls 1
- `/Users/wilson/freshdata-qa/src/freshdata/cleaner.py:189 clean` — self 0.000 s, cumulative 22.016 s, calls 1
- `/Users/wilson/freshdata-qa/src/freshdata/cleaner.py:45 run_pipeline` — self 0.001 s, cumulative 22.016 s, calls 1
- `/Users/wilson/freshdata-qa/src/freshdata/semantic/apply.py:192 run_semantic` — self 0.000 s, cumulative 6.300 s, calls 1
- `/Users/wilson/freshdata-qa/src/freshdata/semantic/context.py:224 build_semantic_context` — self 0.017 s, cumulative 6.184 s, calls 1
- `/Users/wilson/freshdata-qa/src/freshdata/semantic/context.py:68 _build_info` — self 0.321 s, cumulative 5.200 s, calls 32
- `/Users/wilson/freshdata-qa/src/freshdata/steps/dtypes.py:332 fix_dtypes` — self 0.003 s, cumulative 4.842 s, calls 1
- `/Users/wilson/freshdata-qa/src/freshdata/steps/dtypes.py:297 suggest_conversion` — self 0.003 s, cumulative 4.838 s, calls 10
- `~:0 <built-in method builtins.sum>` — self 0.006 s, cumulative 4.240 s, calls 222
- `/Users/wilson/freshdata-qa/src/freshdata/semantic/context.py:43 _share` — self 0.000 s, cumulative 4.232 s, calls 192

### case_id=c4cf49ffc913ad42 rows=500000 width=medium dataset_type=mixed config=default options={"verbose":false} report=false backend=pandas output=pandas seed=42 warmups=1 repetitions=5

- audit_events: 0.0%
- backend_conversion: 0.0%
- context: 0.0%
- correlation: 0.3%
- dtype_repair: 2.0%
- duplicates: 0.4%
- engine_cache: 0.0%
- missing: 0.0%
- outliers: 0.0%
- report_finalization: 0.0%
- role_inference: 0.1%
- semantic_ml: 0.0%
- `/Users/wilson/freshdata-qa/src/freshdata/api.py:138 clean` — self 0.000 s, cumulative 49.320 s, calls 1
- `/Users/wilson/freshdata-qa/src/freshdata/cleaner.py:189 clean` — self 0.000 s, cumulative 49.319 s, calls 1
- `/Users/wilson/freshdata-qa/src/freshdata/cleaner.py:45 run_pipeline` — self 0.002 s, cumulative 49.319 s, calls 1
- `/Users/wilson/freshdata-qa/src/freshdata/steps/duplicates.py:83 drop_duplicate_rows` — self 0.010 s, cumulative 18.036 s, calls 1
- `/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/strings/accessor.py:132 wrapper` — self 0.000 s, cumulative 17.410 s, calls 50
- `/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/string_.py:431 _str_map` — self 0.013 s, cumulative 17.383 s, calls 50
- `/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/string_.py:486 _str_map_str_or_object` — self 5.671 s, cumulative 16.647 s, calls 40
- `/Users/wilson/freshdata-qa/src/freshdata/steps/duplicates.py:70 _filter_rows` — self 0.196 s, cumulative 16.228 s, calls 1
- `/Users/wilson/freshdata-qa/benchmarks/performance/instrumentation.py:58 observed` — self 0.011 s, cumulative 16.063 s, calls 396
- `/Users/wilson/freshdata-qa/src/freshdata/steps/dtypes.py:332 fix_dtypes` — self 0.013 s, cumulative 13.256 s, calls 1

### case_id=52f1c76054406372 rows=1000000 width=narrow dataset_type=mixed config=default options={"verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5

- audit_events: 0.0%
- backend_conversion: 0.0%
- context: 0.0%
- correlation: 0.1%
- dtype_repair: 1.4%
- duplicates: 0.9%
- engine_cache: 0.0%
- missing: 0.0%
- outliers: 0.0%
- report_finalization: 0.0%
- role_inference: 0.1%
- semantic_ml: 0.0%
- `/Users/wilson/freshdata-qa/src/freshdata/api.py:138 clean` — self 0.000 s, cumulative 31.423 s, calls 1
- `/Users/wilson/freshdata-qa/src/freshdata/cleaner.py:189 clean` — self 0.000 s, cumulative 31.422 s, calls 1
- `/Users/wilson/freshdata-qa/src/freshdata/cleaner.py:45 run_pipeline` — self 0.008 s, cumulative 31.422 s, calls 1
- `/Users/wilson/freshdata-qa/src/freshdata/steps/duplicates.py:83 drop_duplicate_rows` — self 0.009 s, cumulative 17.997 s, calls 1
- `/Users/wilson/freshdata-qa/src/freshdata/steps/duplicates.py:70 _filter_rows` — self 0.267 s, cumulative 16.875 s, calls 1
- `/Users/wilson/freshdata-qa/benchmarks/performance/instrumentation.py:58 observed` — self 0.005 s, cumulative 10.112 s, calls 99
- `/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/internals/managers.py:317 apply` — self 0.003 s, cumulative 8.208 s, calls 114
- `/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/generic.py:6485 astype` — self 0.000 s, cumulative 7.929 s, calls 18
- `/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/internals/managers.py:440 astype` — self 0.000 s, cumulative 7.925 s, calls 18
- `/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/internals/blocks.py:749 astype` — self 0.000 s, cumulative 7.925 s, calls 18


## Confirmed root causes

No cause is confirmed by profiling alone; these exact-evidence items are performance candidates:

- `optional_ml_overhead` in `9ea9cb03dfc114c5` (case_id=9ea9cb03dfc114c5 rows=100000 width=medium dataset_type=mixed config=semantic options={"semantic_mode":"assist","verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5): candidate; stage 13.1%, observed calls 1.

## Rejected hypotheses

- `backend_conversion_overhead` in `2e88248a20ec642b` (case_id=2e88248a20ec642b rows=10000 width=wide dataset_type=mixed config=default options={"verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5): insufficient_evidence; stage 0.0%, observed calls 0.
- `copy_pressure` in `2e88248a20ec642b` (case_id=2e88248a20ec642b rows=10000 width=wide dataset_type=mixed config=default options={"verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5): rejected; stage 0.2%, observed calls 101.
- `dtype_conversion_pressure` in `2e88248a20ec642b` (case_id=2e88248a20ec642b rows=10000 width=wide dataset_type=mixed config=default options={"verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5): rejected; stage 2.5%, observed calls 326.
- `optional_ml_overhead` in `2e88248a20ec642b` (case_id=2e88248a20ec642b rows=10000 width=wide dataset_type=mixed config=default options={"verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5): insufficient_evidence; stage 0.0%, observed calls 0.
- `repeated_null_scans` in `2e88248a20ec642b` (case_id=2e88248a20ec642b rows=10000 width=wide dataset_type=mixed config=default options={"verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5): rejected; stage 0.0%, observed calls 1567.
- `repeated_uniqueness_scans` in `2e88248a20ec642b` (case_id=2e88248a20ec642b rows=10000 width=wide dataset_type=mixed config=default options={"verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5): rejected; stage 0.1%, observed calls 256.
- `report_finalization_overhead` in `2e88248a20ec642b` (case_id=2e88248a20ec642b rows=10000 width=wide dataset_type=mixed config=default options={"verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5): rejected; stage 0.0%, observed calls 1.
- `unnecessary_correlation` in `2e88248a20ec642b` (case_id=2e88248a20ec642b rows=10000 width=wide dataset_type=mixed config=default options={"verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5): insufficient_evidence; stage 0.2%, observed calls 1.
- `backend_conversion_overhead` in `52f1c76054406372` (case_id=52f1c76054406372 rows=1000000 width=narrow dataset_type=mixed config=default options={"verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5): insufficient_evidence; stage 0.0%, observed calls 0.
- `copy_pressure` in `52f1c76054406372` (case_id=52f1c76054406372 rows=1000000 width=narrow dataset_type=mixed config=default options={"verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5): rejected; stage 0.0%, observed calls 17.
- `dtype_conversion_pressure` in `52f1c76054406372` (case_id=52f1c76054406372 rows=1000000 width=narrow dataset_type=mixed config=default options={"verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5): rejected; stage 1.4%, observed calls 18.
- `optional_ml_overhead` in `52f1c76054406372` (case_id=52f1c76054406372 rows=1000000 width=narrow dataset_type=mixed config=default options={"verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5): insufficient_evidence; stage 0.0%, observed calls 0.
- `repeated_null_scans` in `52f1c76054406372` (case_id=52f1c76054406372 rows=1000000 width=narrow dataset_type=mixed config=default options={"verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5): rejected; stage 0.0%, observed calls 47.
- `repeated_uniqueness_scans` in `52f1c76054406372` (case_id=52f1c76054406372 rows=1000000 width=narrow dataset_type=mixed config=default options={"verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5): rejected; stage 0.1%, observed calls 16.
- `report_finalization_overhead` in `52f1c76054406372` (case_id=52f1c76054406372 rows=1000000 width=narrow dataset_type=mixed config=default options={"verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5): rejected; stage 0.0%, observed calls 1.
- `unnecessary_correlation` in `52f1c76054406372` (case_id=52f1c76054406372 rows=1000000 width=narrow dataset_type=mixed config=default options={"verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5): insufficient_evidence; stage 0.1%, observed calls 1.
- `backend_conversion_overhead` in `9ea9cb03dfc114c5` (case_id=9ea9cb03dfc114c5 rows=100000 width=medium dataset_type=mixed config=semantic options={"semantic_mode":"assist","verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5): insufficient_evidence; stage 0.0%, observed calls 0.
- `copy_pressure` in `9ea9cb03dfc114c5` (case_id=9ea9cb03dfc114c5 rows=100000 width=medium dataset_type=mixed config=semantic options={"semantic_mode":"assist","verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5): rejected; stage 0.1%, observed calls 61.
- `dtype_conversion_pressure` in `9ea9cb03dfc114c5` (case_id=9ea9cb03dfc114c5 rows=100000 width=medium dataset_type=mixed config=semantic options={"semantic_mode":"assist","verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5): rejected; stage 1.2%, observed calls 108.
- `repeated_null_scans` in `9ea9cb03dfc114c5` (case_id=9ea9cb03dfc114c5 rows=100000 width=medium dataset_type=mixed config=semantic options={"semantic_mode":"assist","verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5): insufficient_evidence; stage 0.0%, observed calls 330.
- `repeated_uniqueness_scans` in `9ea9cb03dfc114c5` (case_id=9ea9cb03dfc114c5 rows=100000 width=medium dataset_type=mixed config=semantic options={"semantic_mode":"assist","verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5): rejected; stage 0.1%, observed calls 139.
- `report_finalization_overhead` in `9ea9cb03dfc114c5` (case_id=9ea9cb03dfc114c5 rows=100000 width=medium dataset_type=mixed config=semantic options={"semantic_mode":"assist","verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5): rejected; stage 0.0%, observed calls 1.
- `unnecessary_correlation` in `9ea9cb03dfc114c5` (case_id=9ea9cb03dfc114c5 rows=100000 width=medium dataset_type=mixed config=semantic options={"semantic_mode":"assist","verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5): insufficient_evidence; stage 0.1%, observed calls 1.
- `backend_conversion_overhead` in `a877f296485e3003` (case_id=a877f296485e3003 rows=100000 width=medium dataset_type=mixed config=aggressive options={"strategy":"aggressive","verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5): insufficient_evidence; stage 0.0%, observed calls 0.
- `copy_pressure` in `a877f296485e3003` (case_id=a877f296485e3003 rows=100000 width=medium dataset_type=mixed config=aggressive options={"strategy":"aggressive","verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5): rejected; stage 0.0%, observed calls 41.
- `dtype_conversion_pressure` in `a877f296485e3003` (case_id=a877f296485e3003 rows=100000 width=medium dataset_type=mixed config=aggressive options={"strategy":"aggressive","verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5): rejected; stage 1.8%, observed calls 73.
- `optional_ml_overhead` in `a877f296485e3003` (case_id=a877f296485e3003 rows=100000 width=medium dataset_type=mixed config=aggressive options={"strategy":"aggressive","verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5): insufficient_evidence; stage 0.0%, observed calls 0.
- `repeated_null_scans` in `a877f296485e3003` (case_id=a877f296485e3003 rows=100000 width=medium dataset_type=mixed config=aggressive options={"strategy":"aggressive","verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5): rejected; stage 0.0%, observed calls 227.
- `repeated_uniqueness_scans` in `a877f296485e3003` (case_id=a877f296485e3003 rows=100000 width=medium dataset_type=mixed config=aggressive options={"strategy":"aggressive","verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5): rejected; stage 0.1%, observed calls 64.
- `report_finalization_overhead` in `a877f296485e3003` (case_id=a877f296485e3003 rows=100000 width=medium dataset_type=mixed config=aggressive options={"strategy":"aggressive","verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5): rejected; stage 0.0%, observed calls 1.
- `unnecessary_correlation` in `a877f296485e3003` (case_id=a877f296485e3003 rows=100000 width=medium dataset_type=mixed config=aggressive options={"strategy":"aggressive","verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5): insufficient_evidence; stage 0.2%, observed calls 1.
- `backend_conversion_overhead` in `c4cf49ffc913ad42` (case_id=c4cf49ffc913ad42 rows=500000 width=medium dataset_type=mixed config=default options={"verbose":false} report=false backend=pandas output=pandas seed=42 warmups=1 repetitions=5): insufficient_evidence; stage 0.0%, observed calls 0.
- `copy_pressure` in `c4cf49ffc913ad42` (case_id=c4cf49ffc913ad42 rows=500000 width=medium dataset_type=mixed config=default options={"verbose":false} report=false backend=pandas output=pandas seed=42 warmups=1 repetitions=5): rejected; stage 0.0%, observed calls 31.
- `dtype_conversion_pressure` in `c4cf49ffc913ad42` (case_id=c4cf49ffc913ad42 rows=500000 width=medium dataset_type=mixed config=default options={"verbose":false} report=false backend=pandas output=pandas seed=42 warmups=1 repetitions=5): rejected; stage 2.0%, observed calls 78.
- `optional_ml_overhead` in `c4cf49ffc913ad42` (case_id=c4cf49ffc913ad42 rows=500000 width=medium dataset_type=mixed config=default options={"verbose":false} report=false backend=pandas output=pandas seed=42 warmups=1 repetitions=5): insufficient_evidence; stage 0.0%, observed calls 0.
- `repeated_null_scans` in `c4cf49ffc913ad42` (case_id=c4cf49ffc913ad42 rows=500000 width=medium dataset_type=mixed config=default options={"verbose":false} report=false backend=pandas output=pandas seed=42 warmups=1 repetitions=5): rejected; stage 0.0%, observed calls 222.
- `repeated_uniqueness_scans` in `c4cf49ffc913ad42` (case_id=c4cf49ffc913ad42 rows=500000 width=medium dataset_type=mixed config=default options={"verbose":false} report=false backend=pandas output=pandas seed=42 warmups=1 repetitions=5): rejected; stage 0.1%, observed calls 64.
- `report_finalization_overhead` in `c4cf49ffc913ad42` (case_id=c4cf49ffc913ad42 rows=500000 width=medium dataset_type=mixed config=default options={"verbose":false} report=false backend=pandas output=pandas seed=42 warmups=1 repetitions=5): rejected; stage 0.0%, observed calls 1.
- `unnecessary_correlation` in `c4cf49ffc913ad42` (case_id=c4cf49ffc913ad42 rows=500000 width=medium dataset_type=mixed config=default options={"verbose":false} report=false backend=pandas output=pandas seed=42 warmups=1 repetitions=5): insufficient_evidence; stage 0.3%, observed calls 1.
- `backend_conversion_overhead` in `eb41d9a6034039f8` (case_id=eb41d9a6034039f8 rows=100000 width=medium dataset_type=mixed config=default options={"verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5): insufficient_evidence; stage 0.0%, observed calls 0.
- `copy_pressure` in `eb41d9a6034039f8` (case_id=eb41d9a6034039f8 rows=100000 width=medium dataset_type=mixed config=default options={"verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5): rejected; stage 0.1%, observed calls 31.
- `dtype_conversion_pressure` in `eb41d9a6034039f8` (case_id=eb41d9a6034039f8 rows=100000 width=medium dataset_type=mixed config=default options={"verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5): rejected; stage 1.6%, observed calls 78.
- `optional_ml_overhead` in `eb41d9a6034039f8` (case_id=eb41d9a6034039f8 rows=100000 width=medium dataset_type=mixed config=default options={"verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5): insufficient_evidence; stage 0.0%, observed calls 0.
- `repeated_null_scans` in `eb41d9a6034039f8` (case_id=eb41d9a6034039f8 rows=100000 width=medium dataset_type=mixed config=default options={"verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5): rejected; stage 0.0%, observed calls 222.
- `repeated_uniqueness_scans` in `eb41d9a6034039f8` (case_id=eb41d9a6034039f8 rows=100000 width=medium dataset_type=mixed config=default options={"verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5): rejected; stage 0.1%, observed calls 64.
- `report_finalization_overhead` in `eb41d9a6034039f8` (case_id=eb41d9a6034039f8 rows=100000 width=medium dataset_type=mixed config=default options={"verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5): rejected; stage 0.0%, observed calls 1.
- `unnecessary_correlation` in `eb41d9a6034039f8` (case_id=eb41d9a6034039f8 rows=100000 width=medium dataset_type=mixed config=default options={"verbose":false} report=true backend=pandas output=pandas seed=42 warmups=1 repetitions=5): insufficient_evidence; stage 0.2%, observed calls 1.

## Failures, timeouts, and OOMs

- 10000 rows / medium / pandas_numeric_median_fill: failed (ChildProcessError: Traceback (most recent call last):
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/internals/blocks.py", line 2401, in fillna
    new_values = self.values.fillna(
                 ^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 267, in fillna
    new_values[mask] = value
    ~~~~~~~~~~^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 315, in __setitem__
    value = self._validate_setitem_value(value)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 306, in _validate_setitem_value
    raise TypeError(f"Invalid value '{value!s}' for dtype '{self.dtype}'")
TypeError: Invalid value '5082.5' for dtype 'Int64'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/Users/wilson/freshdata-qa/benchmarks/performance/baselines.py", line 137, in baseline_worker_main
    result = measure_pandas_baseline(BenchmarkCase(**payload), baseline_name, command=command)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/benchmarks/performance/baselines.py", line 94, in measure_pandas_baseline
    operation(frame)
  File "/Users/wilson/freshdata-qa/benchmarks/performance/baselines.py", line 31, in _numeric_median_fill
    column: frame[column].fillna(frame[column].median())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/generic.py", line 7372, in fillna
    new_data = self._mgr.fillna(
               ^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/internals/base.py", line 186, in fillna
    return self.apply_with_block(
           ^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/internals/managers.py", line 363, in apply
    applied = getattr(b, f)(**kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/internals/blocks.py", line 2407, in fillna
    new_values = self.values.fillna(value=value, method=None, limit=limit)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 267, in fillna
    new_values[mask] = value
    ~~~~~~~~~~^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 315, in __setitem__
    value = self._validate_setitem_value(value)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 306, in _validate_setitem_value
    raise TypeError(f"Invalid value '{value!s}' for dtype '{self.dtype}'")
TypeError: Invalid value '5082.5' for dtype 'Int64'
).
- 10000 rows / narrow / pandas_numeric_median_fill: failed (ChildProcessError: Traceback (most recent call last):
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/internals/blocks.py", line 2401, in fillna
    new_values = self.values.fillna(
                 ^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 267, in fillna
    new_values[mask] = value
    ~~~~~~~~~~^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 315, in __setitem__
    value = self._validate_setitem_value(value)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 306, in _validate_setitem_value
    raise TypeError(f"Invalid value '{value!s}' for dtype '{self.dtype}'")
TypeError: Invalid value '5082.5' for dtype 'Int64'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/Users/wilson/freshdata-qa/benchmarks/performance/baselines.py", line 137, in baseline_worker_main
    result = measure_pandas_baseline(BenchmarkCase(**payload), baseline_name, command=command)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/benchmarks/performance/baselines.py", line 94, in measure_pandas_baseline
    operation(frame)
  File "/Users/wilson/freshdata-qa/benchmarks/performance/baselines.py", line 31, in _numeric_median_fill
    column: frame[column].fillna(frame[column].median())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/generic.py", line 7372, in fillna
    new_data = self._mgr.fillna(
               ^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/internals/base.py", line 186, in fillna
    return self.apply_with_block(
           ^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/internals/managers.py", line 363, in apply
    applied = getattr(b, f)(**kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/internals/blocks.py", line 2407, in fillna
    new_values = self.values.fillna(value=value, method=None, limit=limit)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 267, in fillna
    new_values[mask] = value
    ~~~~~~~~~~^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 315, in __setitem__
    value = self._validate_setitem_value(value)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 306, in _validate_setitem_value
    raise TypeError(f"Invalid value '{value!s}' for dtype '{self.dtype}'")
TypeError: Invalid value '5082.5' for dtype 'Int64'
).
- 10000 rows / wide / pandas_numeric_median_fill: failed (ChildProcessError: Traceback (most recent call last):
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/internals/blocks.py", line 2401, in fillna
    new_values = self.values.fillna(
                 ^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 267, in fillna
    new_values[mask] = value
    ~~~~~~~~~~^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 315, in __setitem__
    value = self._validate_setitem_value(value)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 306, in _validate_setitem_value
    raise TypeError(f"Invalid value '{value!s}' for dtype '{self.dtype}'")
TypeError: Invalid value '5082.5' for dtype 'Int64'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/Users/wilson/freshdata-qa/benchmarks/performance/baselines.py", line 137, in baseline_worker_main
    result = measure_pandas_baseline(BenchmarkCase(**payload), baseline_name, command=command)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/benchmarks/performance/baselines.py", line 94, in measure_pandas_baseline
    operation(frame)
  File "/Users/wilson/freshdata-qa/benchmarks/performance/baselines.py", line 31, in _numeric_median_fill
    column: frame[column].fillna(frame[column].median())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/generic.py", line 7372, in fillna
    new_data = self._mgr.fillna(
               ^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/internals/base.py", line 186, in fillna
    return self.apply_with_block(
           ^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/internals/managers.py", line 363, in apply
    applied = getattr(b, f)(**kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/internals/blocks.py", line 2407, in fillna
    new_values = self.values.fillna(value=value, method=None, limit=limit)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 267, in fillna
    new_values[mask] = value
    ~~~~~~~~~~^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 315, in __setitem__
    value = self._validate_setitem_value(value)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 306, in _validate_setitem_value
    raise TypeError(f"Invalid value '{value!s}' for dtype '{self.dtype}'")
TypeError: Invalid value '5082.5' for dtype 'Int64'
).
- 100000 rows / medium / pandas_numeric_median_fill: failed (ChildProcessError: Traceback (most recent call last):
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/internals/blocks.py", line 2401, in fillna
    new_values = self.values.fillna(
                 ^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 267, in fillna
    new_values[mask] = value
    ~~~~~~~~~~^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 315, in __setitem__
    value = self._validate_setitem_value(value)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 306, in _validate_setitem_value
    raise TypeError(f"Invalid value '{value!s}' for dtype '{self.dtype}'")
TypeError: Invalid value '4989.5' for dtype 'Int64'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/Users/wilson/freshdata-qa/benchmarks/performance/baselines.py", line 137, in baseline_worker_main
    result = measure_pandas_baseline(BenchmarkCase(**payload), baseline_name, command=command)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/benchmarks/performance/baselines.py", line 94, in measure_pandas_baseline
    operation(frame)
  File "/Users/wilson/freshdata-qa/benchmarks/performance/baselines.py", line 31, in _numeric_median_fill
    column: frame[column].fillna(frame[column].median())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/generic.py", line 7372, in fillna
    new_data = self._mgr.fillna(
               ^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/internals/base.py", line 186, in fillna
    return self.apply_with_block(
           ^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/internals/managers.py", line 363, in apply
    applied = getattr(b, f)(**kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/internals/blocks.py", line 2407, in fillna
    new_values = self.values.fillna(value=value, method=None, limit=limit)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 267, in fillna
    new_values[mask] = value
    ~~~~~~~~~~^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 315, in __setitem__
    value = self._validate_setitem_value(value)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 306, in _validate_setitem_value
    raise TypeError(f"Invalid value '{value!s}' for dtype '{self.dtype}'")
TypeError: Invalid value '4989.5' for dtype 'Int64'
).
- 100000 rows / wide / pandas_numeric_median_fill: failed (ChildProcessError: Traceback (most recent call last):
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/internals/blocks.py", line 2401, in fillna
    new_values = self.values.fillna(
                 ^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 267, in fillna
    new_values[mask] = value
    ~~~~~~~~~~^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 315, in __setitem__
    value = self._validate_setitem_value(value)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 306, in _validate_setitem_value
    raise TypeError(f"Invalid value '{value!s}' for dtype '{self.dtype}'")
TypeError: Invalid value '4989.5' for dtype 'Int64'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/Users/wilson/freshdata-qa/benchmarks/performance/baselines.py", line 137, in baseline_worker_main
    result = measure_pandas_baseline(BenchmarkCase(**payload), baseline_name, command=command)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/benchmarks/performance/baselines.py", line 94, in measure_pandas_baseline
    operation(frame)
  File "/Users/wilson/freshdata-qa/benchmarks/performance/baselines.py", line 31, in _numeric_median_fill
    column: frame[column].fillna(frame[column].median())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/generic.py", line 7372, in fillna
    new_data = self._mgr.fillna(
               ^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/internals/base.py", line 186, in fillna
    return self.apply_with_block(
           ^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/internals/managers.py", line 363, in apply
    applied = getattr(b, f)(**kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/internals/blocks.py", line 2407, in fillna
    new_values = self.values.fillna(value=value, method=None, limit=limit)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 267, in fillna
    new_values[mask] = value
    ~~~~~~~~~~^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 315, in __setitem__
    value = self._validate_setitem_value(value)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 306, in _validate_setitem_value
    raise TypeError(f"Invalid value '{value!s}' for dtype '{self.dtype}'")
TypeError: Invalid value '4989.5' for dtype 'Int64'
).
- 500000 rows / medium / pandas_numeric_median_fill: failed (ChildProcessError: Traceback (most recent call last):
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/internals/blocks.py", line 2401, in fillna
    new_values = self.values.fillna(
                 ^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 267, in fillna
    new_values[mask] = value
    ~~~~~~~~~~^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 315, in __setitem__
    value = self._validate_setitem_value(value)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 306, in _validate_setitem_value
    raise TypeError(f"Invalid value '{value!s}' for dtype '{self.dtype}'")
TypeError: Invalid value '4995.5' for dtype 'Int64'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/Users/wilson/freshdata-qa/benchmarks/performance/baselines.py", line 137, in baseline_worker_main
    result = measure_pandas_baseline(BenchmarkCase(**payload), baseline_name, command=command)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/benchmarks/performance/baselines.py", line 94, in measure_pandas_baseline
    operation(frame)
  File "/Users/wilson/freshdata-qa/benchmarks/performance/baselines.py", line 31, in _numeric_median_fill
    column: frame[column].fillna(frame[column].median())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/generic.py", line 7372, in fillna
    new_data = self._mgr.fillna(
               ^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/internals/base.py", line 186, in fillna
    return self.apply_with_block(
           ^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/internals/managers.py", line 363, in apply
    applied = getattr(b, f)(**kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/internals/blocks.py", line 2407, in fillna
    new_values = self.values.fillna(value=value, method=None, limit=limit)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 267, in fillna
    new_values[mask] = value
    ~~~~~~~~~~^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 315, in __setitem__
    value = self._validate_setitem_value(value)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 306, in _validate_setitem_value
    raise TypeError(f"Invalid value '{value!s}' for dtype '{self.dtype}'")
TypeError: Invalid value '4995.5' for dtype 'Int64'
).
- 500000 rows / wide / pandas_numeric_median_fill: failed (ChildProcessError: Traceback (most recent call last):
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/internals/blocks.py", line 2401, in fillna
    new_values = self.values.fillna(
                 ^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 267, in fillna
    new_values[mask] = value
    ~~~~~~~~~~^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 315, in __setitem__
    value = self._validate_setitem_value(value)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 306, in _validate_setitem_value
    raise TypeError(f"Invalid value '{value!s}' for dtype '{self.dtype}'")
TypeError: Invalid value '4995.5' for dtype 'Int64'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/Users/wilson/freshdata-qa/benchmarks/performance/baselines.py", line 137, in baseline_worker_main
    result = measure_pandas_baseline(BenchmarkCase(**payload), baseline_name, command=command)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/benchmarks/performance/baselines.py", line 94, in measure_pandas_baseline
    operation(frame)
  File "/Users/wilson/freshdata-qa/benchmarks/performance/baselines.py", line 31, in _numeric_median_fill
    column: frame[column].fillna(frame[column].median())
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/generic.py", line 7372, in fillna
    new_data = self._mgr.fillna(
               ^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/internals/base.py", line 186, in fillna
    return self.apply_with_block(
           ^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/internals/managers.py", line 363, in apply
    applied = getattr(b, f)(**kwargs)
              ^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/internals/blocks.py", line 2407, in fillna
    new_values = self.values.fillna(value=value, method=None, limit=limit)
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 267, in fillna
    new_values[mask] = value
    ~~~~~~~~~~^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 315, in __setitem__
    value = self._validate_setitem_value(value)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/wilson/freshdata-qa/.venv-qa/lib/python3.12/site-packages/pandas/core/arrays/masked.py", line 306, in _validate_setitem_value
    raise TypeError(f"Invalid value '{value!s}' for dtype '{self.dtype}'")
TypeError: Invalid value '4995.5' for dtype 'Int64'
).

## Limitations

- Component baselines cover only their named pandas operation; no full balanced FreshData pipeline equivalence is claimed.
- Timing classifications require both a 10% effect and twice the larger observed coefficient of variation.
