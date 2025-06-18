| Implementation | Threads | Runtime | Speedup vs Baseline | Efficiency |
|---------------|---------|---------|---------------------| -------------|
| Python Baseline | 1 | 2h 56m 31s (177 mins) | 1.0× | 100% |
| Python OpenMP | 4 | 2h 1m 6s (121 mins)| 1.46× | 36.5% |
| Python OpenMP | 8 | 2h 20m 9s (140 mins) | 1.26x | 15.8% |
| C++ Baseline | 1 | 1h 31m 19s (91 mins) | 1.0× | 100% |
| C++ OpenMP | 4 | 46m 42s | 1.95× | 48.8% |
| C++ OpenMP | 8 | 1h 19 m 4s (79mins) | 1.15x | 14.4% |

# Key Findings/Analysis:

I observed significant differences between Python and C++ implementations of the NMF algorithm, as well as the effects of parallelization using OpenMP. The C++ baseline implementation already demonstrated a substantial performance advantage over Python, with almost a 2x speedup by language optimization alone. Both implementations showed similar patterns of diminishing returns with increased thread counts. This  might suggest that the NMF algorithm implementation encounters resource contention, memory bandwidth limitations, or load balancing issues that prevent effective utilization of higher thread counts, indicating that 4 threads represents the sweet spot for this particular workload and system configuration. 
