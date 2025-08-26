# CBFtorch Benchmarking - Quick Start Guide

## ✅ What's Working Now

The benchmarking infrastructure is fully operational and saves detailed results in JSON format.

## 🚀 Quick Commands

### 1. Run Simple Comprehensive Benchmark
```bash
python benchmarks/simple_benchmark.py
```
- **Output**: `benchmarks/results/performance_results_TIMESTAMP.json`
- **Summary**: `benchmarks/results/performance_summary_TIMESTAMP.json`
- Tests barrier evaluation, HOCBF computation, and Lie derivatives
- Includes memory usage tracking and performance analysis

### 2. Analyze Results
```bash
# Text summary
python benchmarks/analyze_results.py --results-dir benchmarks/results --summary

# Generate plots and HTML report
python benchmarks/analyze_results.py --results-dir benchmarks/results --plot --report --output-dir benchmarks/analysis
```

### 3. Create Simple Baseline (Legacy)
```bash
python benchmarks/create_baseline.py
```

## 📊 What Gets Saved

### Detailed Results (`performance_results_TIMESTAMP.json`)
```json
{
  "metadata": {
    "timestamp": "2025-07-11 11:57:11",
    "batch_sizes": [1, 5, 10, 25, 50, 100],
    "num_runs": 50,
    "torch_version": "2.7.1+cu126",
    "device": "cuda"
  },
  "barrier_evaluation": {
    "rel_deg_1_batch_1": {
      "mean_time": 7.468e-05,
      "std_time": 3.456e-05,
      "memory_delta_mb": 113.45,
      "batch_size": 1,
      "rel_deg": 1,
      "time_per_sample": 7.468e-05
    }
  },
  "hocbf_evaluation": { /* ... */ },
  "lie_derivatives": { /* ... */ },
  "memory_analysis": { /* ... */ }
}
```

### Summary Results (`performance_summary_TIMESTAMP.json`)
```json
{
  "overview": {
    "total_tests": 36,
    "avg_barrier_time": 3.813e-05,
    "avg_hocbf_time": 0.000377,
    "max_execution_time": 0.001293
  },
  "scaling_analysis": {
    "batch_1_to_100_scaling": 1.33,
    "scaling_efficiency": 75.4
  },
  "performance_hotspots": [
    ["rel_deg_3_batch_100", 0.001293],
    ["rel_deg_3_batch_25", 0.000926]
  ]
}
```

## 🎯 Key Insights from Current Baseline

### Performance Hotspots
1. **Relative degree 3 operations** are 10-40x slower than relative degree 1
2. **HOCBF evaluation** is ~10x slower than basic barrier evaluation
3. **Lie derivatives** are the most expensive operations (>1ms for rel_deg=2)

### Memory Usage
- Initial memory allocation: ~113MB (first test)
- Subsequent tests: minimal memory growth (~0-5MB)
- Total benchmark memory footprint: ~120MB

### Scaling Analysis
- **Batch scaling efficiency**: 75.4% (good vectorization)
- **Per-sample time**: decreases with larger batches (as expected)

## 🔧 Optimization Targets Identified

1. **High Impact**: 
   - Lie derivative computation (currently 1-2ms for complex cases)
   - Higher-order CBF evaluation (exponential growth with relative degree)

2. **Medium Impact**:
   - Gradient computation redundancy
   - Memory allocation patterns

3. **Lower Impact**:
   - Basic barrier evaluation (already well optimized)

## 📁 File Organization

- `benchmarks/results/` - All JSON result files
- `benchmarks/analysis/` - Generated plots and HTML reports  
- `benchmarks/` - Individual baseline JSON files (legacy)

## 🔄 Optimization Workflow

1. **Create baseline**: `python benchmarks/simple_benchmark.py`
2. **Implement optimizations** in cbftorch code
3. **Re-run benchmark**: `python benchmarks/simple_benchmark.py` 
4. **Compare results**: Use the analysis tools to measure improvements
5. **Validate correctness**: (Future: use full test suite)

The framework is ready for optimization development!