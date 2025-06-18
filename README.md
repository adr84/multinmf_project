# MultiNMF Acceleration: GPU & MPI Implementation
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![MPI](https://img.shields.io/badge/MPI-OpenMPI-green.svg)](https://www.open-mpi.org/)

## Overview
This project implements and compares three acceleration strategies for Multi-view Non-negative Matrix Factorization (MultiNMF). 

## Repository Navigation

### Core Implementation
- 📊 [**Baseline Performance**](part0-baseline/) - Python & C++ baseline testing
- ⚡ [**CPU Acceleration**](part1-openmp/) - OpenMP parallelization results  
- 🚀 [**GPU Acceleration**](part2-gpu/) - Apple M1 MPS implementation
- 🌐 [**Distributed Computing**](part3-mpi/) - MPI parallel processing

### Analysis & Results
- 📈 [**Performance Summary**](results/) - Cross-method comparisons
- 📋 [**Technical Documentation**](docs/) - Complete analysis and specifications

### Quick Links
- [System Specifications](docs/system_specifications.md)
- [Performance Results](results/performance_summary.md)
- [Assignment Deliverables](docs/assignment_deliverables.md)
