# CBFtorch Benchmarking

**Simple, clean benchmarking for CBFtorch optimization.**

## 🚀 Quick Start

### Run Benchmark
```bash
# Full benchmark (recommended)
python benchmarks/benchmark.py

# Quick test (faster, fewer runs)  
python benchmarks/benchmark.py --quick

# Baseline only (skip optimization analysis)
python benchmarks/benchmark.py --baseline
```

### Analyze Results
```bash
# Analyze latest results
python benchmarks/analyze.py

# Generate plots
python benchmarks/analyze.py --plot

# Compare two benchmark runs
python benchmarks/analyze.py --compare file1.json file2.json
```

## 📊 What Gets Measured

### Basic Performance
- **Barrier evaluation** across different relative degrees (1, 2, 3)
- **HOCBF computation** with scaling analysis
- **Batch size performance** (1, 10, 50, 100 samples)
- **Memory usage** tracking

### Optimization Targets  
- **Gradient computation redundancy** (62.9% potential savings identified)
- **Multiple barriers scaling** (composite barrier performance)
- **Repeated computations** (simulation-like patterns)

## 📁 File Organization

```
benchmarks/
├── benchmark.py              # ← Main benchmark (ONLY file you need!)
├── analyze.py               # ← Results analysis and plotting
├── results/                 # ← All benchmark data saved here
│   └── cbftorch_benchmark_*.json
├── analysis/                # ← Generated plots and reports
├── scripts/                 # ← Old/complex scripts (ignore)
└── [other folders]          # ← Framework code (ignore)
```

## 🎯 Optimization Workflow

1. **Create baseline**: `python benchmarks/benchmark.py`
2. **Implement optimizations** in your code
3. **Re-run benchmark**: `python benchmarks/benchmark.py`  
4. **Compare results**: `python benchmarks/analyze.py --compare baseline.json optimized.json`
5. **Validate improvements** and iterate

## 📈 Key Insights from Current Baseline

- **HOCBF relative degree 3**: 40x slower than relative degree 1
- **Gradient computation**: 2.9x inefficiency factor (major optimization target)
- **Memory efficiency**: Excellent (no leaks detected)
- **Batch scaling**: 75% efficiency (good vectorization)

## 🔧 Two Files, That's It!

- **`benchmark.py`** - Run this to measure performance
- **`analyze.py`** - Run this to understand results

Everything else is support code or old versions. **You only need these two files.**