# MultiNMF Acceleration: GPU & MPI Implementation
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)
[![MPI](https://img.shields.io/badge/MPI-OpenMPI-green.svg)](https://www.open-mpi.org/)

## Overview
This project implements and compares three acceleration strategies for Multi-view Non-negative Matrix Factorization (MultiNMF). 

multinmf-acceleration/
├── README.md                           # Main overview (executive summary)
├── docs/
│   ├── system_specifications.md        # All system specs
│   ├── complete_analysis.md           # Full technical analysis
│   └── assignment_deliverables.md     # Final deliverable summary
├── part0-baseline/
│   ├── README.md                      # Part 0 overview
│   └── baseline_results.md            # Part 0 detailed results
├── part1-openmp/
│   ├── README.md                      # Part 1 overview
│   └── openmp_results.md              # Part 1 detailed results
├── part2-gpu/
│   ├── README.md                      # Part 2 overview
│   ├── gpu_implementation.md          # Technical implementation details
│   └── gpu_results.md                 # Part 2 detailed results
├── part3-mpi/
│   ├── README.md                      # Part 3 overview
│   └── mpi_results.md                 # Part 3 results
└── results/
    └── performance_summary.md         # Cross-part performance comparison
