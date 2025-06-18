# MultiNMF Acceleration: GPU & MPI Implementation
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![MPI](https://img.shields.io/badge/MPI-OpenMPI-green.svg)](https://www.open-mpi.org/)

## Overview
This project implements and compares three acceleration strategies for Multi-view Non-negative Matrix Factorization (MultiNMF). 

## Repository Structure

- **README.md** - Main project overview and executive summary
- **docs/** - Documentation and analysis
  - system_specifications.md - Complete system specifications
  - complete_analysis.md - Full technical analysis
  - assignment_deliverables.md - Final deliverable summary
- **part0-baseline/** - Baseline performance testing
  - README.md - Part 0 overview
  - baseline_results.md - Detailed baseline results
- **part1-openmp/** - CPU parallelization with OpenMP
  - README.md - Part 1 overview  
  - openmp_results.md - OpenMP performance results
- **part2-gpu/** - GPU acceleration implementation
  - README.md - Part 2 overview
  - gpu_implementation.md - Technical implementation details
  - gpu_results.md - GPU performance results
- **part3-mpi/** - MPI distributed computing
  - README.md - Part 3 overview
  - mpi_results.md - MPI performance results
- **results/** - Performance comparisons and visualizations
  - performance_summary.md - Cross-method performance analysis
