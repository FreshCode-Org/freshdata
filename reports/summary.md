# FreshData Benchmark Results

## BenchmarkPipeline
### `peakmem_full_clean`
| Library     | 10000 rows   | 100000 rows   |
|-------------|--------------|---------------|
| 'freshdata' | 131.3 MB     | -             |
| 'pandas'    | 159.9 MB     | 193.3 MB      |
| 'polars'    | -            | 321.7 MB      |
| 'pyjanitor' | -            | 383.8 MB      |
| 'autoclean' | 356.4 MB     | -             |

### `time_full_clean`
| Library     | 10000 rows                       | 100000 rows                      |
|-------------|----------------------------------|----------------------------------|
| 'freshdata' | 463.3 ms (IQR 463.3 ms-463.3 ms) | -                                |
| 'pandas'    | 56.7 ms (IQR 56.7 ms-56.7 ms)    | 22.3 ms (IQR 22.3 ms-22.3 ms)    |
| 'polars'    | -                                | 3.29 s (IQR 3.29 s-3.29 s)       |
| 'pyjanitor' | -                                | 438.6 ms (IQR 438.6 ms-438.6 ms) |
| 'autoclean' | 174.0 ms (IQR 174.1 ms-174.1 ms) | -                                |

### `track_output_cols`
| Library     | 10000 rows   | 100000 rows   |
|-------------|--------------|---------------|
| 'freshdata' | 19           | -             |
| 'pandas'    | 19           | 19            |
| 'polars'    | -            | 19            |
| 'pyjanitor' | -            | 19            |
| 'autoclean' | 19           | -             |

### `track_output_rows`
| Library     | 10000 rows   | 100000 rows   |
|-------------|--------------|---------------|
| 'freshdata' | 10000        | -             |
| 'pandas'    | 10000        | 10000         |
| 'polars'    | -            | 100000        |
| 'pyjanitor' | -            | 100000        |
| 'autoclean' | 100000       | -             |

### `track_throughput`
| Library     | 10000 rows   | 100000 rows   |
|-------------|--------------|---------------|
| 'freshdata' | 21.5K/s      | -             |
| 'pandas'    | 181.8K/s     | 456.0K/s      |
| 'polars'    | -            | 30.0K/s       |
| 'pyjanitor' | -            | 230.5K/s      |
| 'autoclean' | 376.6K/s     | -             |

