# FreshData Benchmark Results

## BenchmarkFreshDataVsPandas
### `peakmem_detect_outliers`
| Library     | 10000 rows   | 100000 rows   | 1000000 rows   |
|-------------|--------------|---------------|----------------|
| 'freshdata' | 137.5 MB     | 335.9 MB      | 1.78 GB        |
| 'pandas'    | 112.7 MB     | 204.8 MB      | 1020.7 MB      |

### `peakmem_full_pipeline`
| Library     | 10000 rows   | 100000 rows   | 1000000 rows   |
|-------------|--------------|---------------|----------------|
| 'freshdata' | 128.0 MB     | 320.1 MB      | 1.44 GB        |
| 'pandas'    | 123.3 MB     | 272.2 MB      | 1.53 GB        |

### `peakmem_group_aggregations`
| Library     | 10000 rows   | 100000 rows   | 1000000 rows   |
|-------------|--------------|---------------|----------------|
| 'freshdata' | 122.2 MB     | 282.5 MB      | 1.20 GB        |
| 'pandas'    | 112.0 MB     | 200.8 MB      | 991.9 MB       |

### `peakmem_handle_missing`
| Library     | 10000 rows   | 100000 rows   | 1000000 rows   |
|-------------|--------------|---------------|----------------|
| 'freshdata' | 209.9 MB     | 398.5 MB      | 1.54 GB        |
| 'pandas'    | 109.7 MB     | 203.3 MB      | 1008.1 MB      |

### `peakmem_load_csv`
| Library     | 10000 rows   | 100000 rows   | 1000000 rows   |
|-------------|--------------|---------------|----------------|
| 'freshdata' | 136.6 MB     | 315.2 MB      | 1.66 GB        |
| 'pandas'    | 124.1 MB     | 259.5 MB      | 1.56 GB        |

### `peakmem_resolve_duplicates`
| Library     | 10000 rows   | 100000 rows   | 1000000 rows   |
|-------------|--------------|---------------|----------------|
| 'freshdata' | 124.9 MB     | 295.2 MB      | 1.40 GB        |
| 'pandas'    | 115.9 MB     | 248.3 MB      | 1.12 GB        |

### `time_detect_outliers`
| Library     | 10000 rows                       | 100000 rows                      | 1000000 rows                  |
|-------------|----------------------------------|----------------------------------|-------------------------------|
| 'freshdata' | 533.7 ms (IQR 533.7 ms-533.7 ms) | 3.50 s (IQR 3.50 s-3.50 s)       | 31.83 s (IQR 31.83 s-31.83 s) |
| 'pandas'    | 33.6 ms (IQR 33.6 ms-33.6 ms)    | 292.1 ms (IQR 292.1 ms-292.1 ms) | 2.45 s (IQR 2.45 s-2.45 s)    |

### `time_full_pipeline`
| Library     | 10000 rows                       | 100000 rows                      | 1000000 rows                  |
|-------------|----------------------------------|----------------------------------|-------------------------------|
| 'freshdata' | 460.8 ms (IQR 460.8 ms-460.8 ms) | 3.29 s (IQR 3.29 s-3.29 s)       | 31.54 s (IQR 31.54 s-31.54 s) |
| 'pandas'    | 67.8 ms (IQR 67.8 ms-67.8 ms)    | 516.3 ms (IQR 516.3 ms-516.3 ms) | 5.42 s (IQR 5.42 s-5.42 s)    |

### `time_group_aggregations`
| Library     | 10000 rows                       | 100000 rows                      | 1000000 rows                  |
|-------------|----------------------------------|----------------------------------|-------------------------------|
| 'freshdata' | 364.5 ms (IQR 364.4 ms-364.4 ms) | 2.60 s (IQR 2.60 s-2.60 s)       | 24.69 s (IQR 24.69 s-24.69 s) |
| 'pandas'    | 29.2 ms (IQR 29.2 ms-29.2 ms)    | 239.7 ms (IQR 239.7 ms-239.7 ms) | 2.39 s (IQR 2.39 s-2.39 s)    |

### `time_handle_missing`
| Library     | 10000 rows                    | 100000 rows                      | 1000000 rows                  |
|-------------|-------------------------------|----------------------------------|-------------------------------|
| 'freshdata' | 1.92 s (IQR 1.92 s-1.92 s)    | 4.50 s (IQR 4.50 s-4.50 s)       | 33.39 s (IQR 33.39 s-33.39 s) |
| 'pandas'    | 22.8 ms (IQR 22.8 ms-22.8 ms) | 212.2 ms (IQR 212.2 ms-212.2 ms) | 2.13 s (IQR 2.13 s-2.13 s)    |

### `time_load_csv`
| Library     | 10000 rows                       | 100000 rows                      | 1000000 rows                  |
|-------------|----------------------------------|----------------------------------|-------------------------------|
| 'freshdata' | 369.3 ms (IQR 369.3 ms-369.3 ms) | 2.70 s (IQR 2.70 s-2.70 s)       | 26.41 s (IQR 26.41 s-26.41 s) |
| 'pandas'    | 45.9 ms (IQR 45.9 ms-45.9 ms)    | 426.9 ms (IQR 426.9 ms-426.9 ms) | 4.32 s (IQR 4.32 s-4.32 s)    |

### `time_resolve_duplicates`
| Library     | 10000 rows                       | 100000 rows                      | 1000000 rows                  |
|-------------|----------------------------------|----------------------------------|-------------------------------|
| 'freshdata' | 365.5 ms (IQR 365.5 ms-365.5 ms) | 2.76 s (IQR 2.76 s-2.76 s)       | 27.70 s (IQR 27.70 s-27.70 s) |
| 'pandas'    | 17.2 ms (IQR 17.2 ms-17.2 ms)    | 140.4 ms (IQR 140.4 ms-140.4 ms) | 1.90 s (IQR 1.90 s-1.90 s)    |

### `track_pipeline_throughput`
| Library     | 10000 rows   | 100000 rows   | 1000000 rows   |
|-------------|--------------|---------------|----------------|
| 'freshdata' | 18.9K/s      | 24.9K/s       | 20.6K/s        |
| 'pandas'    | 127.2K/s     | 163.1K/s      | 129.4K/s       |

## BenchmarkGroupAgg
### `peakmem_group_agg_multi`
| Library     | 10000 rows   | 100000 rows   | 1000000 rows   |
|-------------|--------------|---------------|----------------|
| 'freshdata' | 122.7 MB     | 287.9 MB      | 1.26 GB        |
| 'pandas'    | 111.6 MB     | 204.5 MB      | 978.3 MB       |

### `time_group_agg_multi`
| Library     | 10000 rows                       | 100000 rows                      | 1000000 rows                  |
|-------------|----------------------------------|----------------------------------|-------------------------------|
| 'freshdata' | 759.6 ms (IQR 759.6 ms-759.6 ms) | 5.81 s (IQR 5.81 s-5.81 s)       | 52.63 s (IQR 52.63 s-52.63 s) |
| 'pandas'    | 72.4 ms (IQR 72.4 ms-72.4 ms)    | 456.3 ms (IQR 456.3 ms-456.3 ms) | 4.55 s (IQR 4.55 s-4.55 s)    |

### `time_group_agg_single`
| Library     | 10000 rows                       | 100000 rows                   | 1000000 rows                     |
|-------------|----------------------------------|-------------------------------|----------------------------------|
| 'freshdata' | 981.8 ms (IQR 981.8 ms-981.8 ms) | 6.11 s (IQR 6.11 s-6.11 s)    | 42.98 s (IQR 42.98 s-42.98 s)    |
| 'pandas'    | 12.7 ms (IQR 12.7 ms-12.7 ms)    | 35.2 ms (IQR 35.2 ms-35.2 ms) | 312.3 ms (IQR 312.3 ms-312.3 ms) |

### `time_group_agg_transform`
| Library     | 10000 rows                       | 100000 rows                      | 1000000 rows                  |
|-------------|----------------------------------|----------------------------------|-------------------------------|
| 'freshdata' | 544.3 ms (IQR 544.3 ms-544.3 ms) | 5.01 s (IQR 5.01 s-5.01 s)       | 43.06 s (IQR 43.06 s-43.06 s) |
| 'pandas'    | 36.9 ms (IQR 36.9 ms-36.9 ms)    | 324.3 ms (IQR 324.3 ms-324.3 ms) | 3.20 s (IQR 3.20 s-3.20 s)    |

### `track_throughput_group_agg`
| Library     | 10000 rows   | 100000 rows   | 1000000 rows   |
|-------------|--------------|---------------|----------------|
| 'freshdata' | 14.0K/s      | 25.4K/s       | 33.9K/s        |
| 'pandas'    | 295.1K/s     | 180.4K/s      | 353.0K/s       |

