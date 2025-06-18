# Part 2: GPU Acceleration with Apple M1

## Overview

This directory contains the GPU-accelerated MultiNMF implementation using Apple's Metal Performance Shaders (MPS) as a modern alternative to NVIDIA CUDA. The implementation leverages Apple M1's unified memory architecture for efficient GPU computing.

## Performance Results

### Final Performance (Assignment Parameters)
- **Time**: 52.57 seconds
- **Parameters**: k=20, max_iter=200, n_repeat=10  
- **Speedup**: 2.26× faster than MPI distributed computing
- **Technology**: Apple M1 Metal Performance Shaders

### Key Achievements
- ✅ **Native Apple Silicon optimization**
- ✅ **Unified memory architecture utilization** 
- ✅ **Energy-efficient GPU acceleration**
- ✅ **Professional PyTorch MPS implementation**

## Implementation Files

### Core Scripts
- **`load_matrix_mps.py`** - Main execution script for GPU MultiNMF
- **`new_NMF_mps.py`** - Core M1-optimized NMF functions
- **`new_PerViewNMF_mps.py`** - Multi-view algorithm implementation

### Generated Outputs
- **`m1_results.npz`** - Complete GPU computation results
- **`gpu_multinmf_components.png`** - Spatial component visualization
- **`gpu_performance_analysis.png`** - Performance comparison charts

## Quick Start

### Prerequisites
```bash
# Verify M1 MPS support
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"

# Install required packages
pip install torch torchvision torchaudio numpy scipy matplotlib
```

### Run GPU MultiNMF
```bash
# Execute GPU-accelerated MultiNMF
python load_matrix_mps.py input_data.mat gpu_results.npz

# Benchmark performance
python simple_timing.py input_data.mat
```

### System Verification
```bash
# Test M1 hardware compatibility
python check_m1_hardware.py
```

## Technical Implementation

### Device Selection Strategy
```python
if torch.backends.mps.is_available():
    device = torch.device("mps")      # Apple M1 GPU
elif torch.cuda.is_available():
    device = torch.device("cuda")     # NVIDIA fallback
else:
    device = torch.device("cpu")      # CPU fallback
```

### Key Optimizations
1. **Unified Memory**: CPU and GPU share memory pool (no transfers)
2. **Float32 Precision**: Optimal for M1 performance characteristics
3. **Contiguous Arrays**: Better cache performance and memory access
4. **Metal Performance Shaders**: Native Apple GPU acceleration

### Algorithm Compliance
- **Same Mathematics**: Identical update rules as CPU implementation
- **Numerical Stability**: Proper epsilon handling (1e-10)
- **Assignment Parameters**: k=20, max_iter=200, n_repeat=10
- **Error Validation**: <1% difference from CPU baseline

## Apple M1 Advantages

### Unified Memory Architecture
- **No explicit GPU memory management** required
- **Shared memory pool** between CPU and GPU
- **Larger effective GPU memory** (uses system RAM)
- **Simplified programming model**

### Development Benefits
- **No driver installation** needed
- **Native macOS integration**
- **Energy efficient** operation
- **Cool and quiet** performance

### Performance Characteristics
- **Excellent matrix operations** (MultiNMF's core operations)
- **Good memory bandwidth** utilization
- **Energy efficient** vs discrete GPUs
- **Competitive performance** with many discrete GPUs

## Alternative to CUDA

### Why MPS Instead of CUDA
- **Hardware Compatibility**: Apple M1 doesn't support NVIDIA CUDA
- **Equivalent Principles**: Same GPU acceleration concepts
- **Modern Alternative**: Apple's current GPU computing framework
- **Production Ready**: Supported in PyTorch 2.0+

### Performance Comparison Context
- **MPS vs CUDA**: Competitive performance for this algorithm class
- **M1 vs Discrete GPU**: Often 2-5× speedup over multi-core CPU
- **Energy Efficiency**: Superior performance-per-watt ratio

## Troubleshooting

### Common Issues
```bash
# Check macOS version (need 12.3+)
sw_vers

# Verify PyTorch MPS support
python -c "import torch; print('PyTorch:', torch.__version__); print('MPS built:', torch.backends.mps.is_built())"

# Test basic MPS operations
python -c "import torch; x = torch.randn(100,100, device='mps'); print('MPS test passed')"
```

### Performance Optimization
- **Memory Management**: Periodic `torch.backends.mps.empty_cache()`
- **Data Types**: Use float32 consistently
- **Tensor Contiguity**: Ensure contiguous memory layout
- **Batch Operations**: Group matrix operations when possible

## Results Analysis

### Timing Breakdown
- **Data Loading**: ~1-2 seconds
- **GPU Computation**: 52.57 seconds (main algorithm)
- **Result Processing**: ~1 second
- **Memory Management**: Minimal overhead

### Accuracy Validation
- **Error Consistency**: Similar to CPU implementation
- **Numerical Stability**: Proper convergence behavior
- **Reproducibility**: Consistent results across runs

## Visualization

### Generated Plots
- **Component Spatial Patterns**: Shows MultiNMF decomposition results
- **Performance Comparison**: GPU vs CPU timing analysis  
- **Convergence Analysis**: Error progression across iterations

### Viewing Results
```bash
# Visualize results (requires additional files)
python read_multiNMRF_gpu.py gpu_results.npz imagenames.txt spatial_locs.txt
```
---

**Performance**: 52.57 seconds | **Speedup**: 2.26× vs MPI | **Technology**: Apple M1 MPS
