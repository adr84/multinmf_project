import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from scipy.io import loadmat

def load_data_safely(filename):
    """Load and inspect the data file"""
    print(f"Loading data from {filename}...")
    
    data = loadmat(filename)
    
    keys = [k for k in data.keys() if not k.startswith('__')]
    print(f"Available keys: {keys}")
    
    matrices = {}
    for key in keys:
        matrix = data[key]
        if hasattr(matrix, 'shape') and len(matrix.shape) == 2:
            matrices[key] = matrix
            print(f"Matrix '{key}': shape {matrix.shape}, dtype {matrix.dtype}")
    
    if len(matrices) >= 2:
        sorted_matrices = sorted(matrices.items(), key=lambda x: x[1].size, reverse=True)
        A_key, A = sorted_matrices[0]
        B_key, B = sorted_matrices[1]
        
        print(f"\nSelected matrices:")
        print(f"A ('{A_key}'): {A.shape}")
        print(f"B ('{B_key}'): {B.shape}")
        
        min_cols = min(A.shape[1], B.shape[1])
        A = A[:, :min_cols].astype(np.float32)
        B = B[:, :min_cols].astype(np.float32)
        
        print(f"After dimension matching:")
        print(f"A: {A.shape}")
        print(f"B: {B.shape}")
        
        return A, B
    else:
        raise ValueError("Could not find two suitable matrices in the file")

def simple_nmf_cpu(A, B, k=20, n_iter=50, n_threads=8):
    """Simple CPU NMF implementation for timing"""
    
    os.environ["OMP_NUM_THREADS"] = str(n_threads)
    os.environ["MKL_NUM_THREADS"] = str(n_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)
    
    print(f"Running CPU NMF (k={k}, iter={n_iter}, threads={n_threads})...")
    
    n_A, m = A.shape
    t_B, _ = B.shape
    
    np.random.seed(42)
    U = np.random.rand(m, k).astype(np.float32) + 0.1
    V_A = np.random.rand(n_A, k).astype(np.float32) + 0.1
    V_B = np.random.rand(t_B, k).astype(np.float32) + 0.1
    
    start_time = time.time()
    
    for i in range(n_iter):
        V_A = V_A * (A @ U) / (V_A @ (U.T @ U) + 1e-10)
        
        V_B = V_B * (B @ U) / (V_B @ (U.T @ U) + 1e-10)
        
        U = U * (A.T @ V_A + B.T @ V_B) / (U @ (V_A.T @ V_A) + U @ (V_B.T @ V_B) + 1e-10)
    
    cpu_time = time.time() - start_time
    
    error_A = np.linalg.norm(A - V_A @ U.T, 'fro') ** 2
    error_B = np.linalg.norm(B - V_B @ U.T, 'fro') ** 2
    total_error = error_A + error_B
    
    print(f"CPU completed in {cpu_time:.2f} seconds, error: {total_error:.6f}")
    
    return cpu_time, total_error

def simple_nmf_m1(A, B, k=20, n_iter=50):
    """Simple M1 GPU NMF implementation for timing"""
    
    if not torch.backends.mps.is_available():
        print("M1 GPU not available")
        return None, None
    
    print(f"Running M1 GPU NMF (k={k}, iter={n_iter})...")
    
    device = torch.device("mps")
    

    A_gpu = torch.tensor(A, device=device, dtype=torch.float32)
    B_gpu = torch.tensor(B, device=device, dtype=torch.float32)
    
    n_A, m = A_gpu.shape
    t_B, _ = B_gpu.shape
    

    torch.manual_seed(42)
    U = torch.rand(m, k, device=device, dtype=torch.float32) + 0.1
    V_A = torch.rand(n_A, k, device=device, dtype=torch.float32) + 0.1
    V_B = torch.rand(t_B, k, device=device, dtype=torch.float32) + 0.1
    
    start_time = time.time()
    
    for i in range(n_iter):
        V_A = V_A * torch.mm(A_gpu, U) / (torch.mm(V_A, torch.mm(U.T, U)) + 1e-10)
        
        V_B = V_B * torch.mm(B_gpu, U) / (torch.mm(V_B, torch.mm(U.T, U)) + 1e-10)
        
        numerator = torch.mm(A_gpu.T, V_A) + torch.mm(B_gpu.T, V_B)
        denominator = torch.mm(U, torch.mm(V_A.T, V_A)) + torch.mm(U, torch.mm(V_B.T, V_B)) + 1e-10
        U = U * (numerator / denominator)
    
    if hasattr(torch, 'mps') and hasattr(torch.mps, 'synchronize'):
        torch.mps.synchronize()
    
    m1_time = time.time() - start_time
    
    error_A = torch.norm(A_gpu - torch.mm(V_A, U.T), p='fro').item() ** 2
    error_B = torch.norm(B_gpu - torch.mm(V_B, U.T), p='fro').item() ** 2
    total_error = error_A + error_B
    
    print(f"M1 GPU completed in {m1_time:.2f} seconds, error: {total_error:.6f}")
    
    del A_gpu, B_gpu, U, V_A, V_B
    if hasattr(torch.backends.mps, 'empty_cache'):
        torch.backends.mps.empty_cache()
    
    return m1_time, total_error

def run_timing_comparison(filename, n_runs=3):
    """Run complete timing comparison"""
    
    print("="*60)
    print("CPU vs M1 GPU MultiNMF Timing Comparison")
    print("="*60)
    
    A, B = load_data_safely(filename)
    
    print(f"\nRunning {n_runs} trials of each method...")
    
    print(f"\n{'='*30}")
    print("CPU TESTS")
    print("="*30)
    
    cpu_times = []
    cpu_errors = []
    
    for run in range(n_runs):
        print(f"\nCPU Run {run + 1}/{n_runs}:")
        cpu_time, cpu_error = simple_nmf_cpu(A, B)
        cpu_times.append(cpu_time)
        cpu_errors.append(cpu_error)
    
    print(f"\n{'='*30}")
    print("M1 GPU TESTS")
    print("="*30)
    
    m1_times = []
    m1_errors = []
    
    for run in range(n_runs):
        print(f"\nM1 GPU Run {run + 1}/{n_runs}:")
        m1_time, m1_error = simple_nmf_m1(A, B)
        if m1_time is not None:
            m1_times.append(m1_time)
            m1_errors.append(m1_error)
    
    print(f"\n{'='*60}")
    print("RESULTS ANALYSIS")
    print("="*60)
    
    if cpu_times:
        cpu_avg = np.mean(cpu_times)
        cpu_std = np.std(cpu_times)
        print(f"CPU (8 threads):     {cpu_avg:7.2f} ± {cpu_std:5.2f} seconds")
        print(f"CPU error average:   {np.mean(cpu_errors):.6f}")
    
    if m1_times:
        m1_avg = np.mean(m1_times)
        m1_std = np.std(m1_times)
        print(f"M1 GPU:             {m1_avg:7.2f} ± {m1_std:5.2f} seconds")
        print(f"M1 error average:    {np.mean(m1_errors):.6f}")
        
        if cpu_times:
            speedup = cpu_avg / m1_avg
            time_saved = cpu_avg - m1_avg
            print(f"\nSpeedup:            {speedup:7.2f}x")
            print(f"Time saved:         {time_saved:7.2f} seconds ({(1-m1_avg/cpu_avg)*100:.1f}%)")
    
    if cpu_times and m1_times:
        plt.figure(figsize=(10, 6))
        
        methods = ['CPU\n(8 threads)', 'M1 GPU']
        times = [cpu_avg, m1_avg]
        errors = [cpu_std, m1_std]
        colors = ['lightblue', 'orange']
        
        bars = plt.bar(methods, times, yerr=errors, capsize=5, 
                      color=colors, alpha=0.7, edgecolor='black')
        
        plt.ylabel('Time (seconds)')
        plt.title('MultiNMF Performance Comparison')
        plt.grid(True, alpha=0.3)
        
        for bar, time_val in zip(bars, times):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + max(errors)*0.1,
                    f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        speedup = times[0] / times[1]
        plt.text(0.5, max(times) * 0.8, f'{speedup:.1f}x\nspeedup', 
                ha='center', va='center', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig('simple_timing_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved: simple_timing_comparison.png")
        plt.show()
    
    return {
        'cpu_times': cpu_times,
        'cpu_errors': cpu_errors,
        'm1_times': m1_times,
        'm1_errors': m1_errors
    }

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 simple_timing.py <input_file.mat>")
        print("Example: python3 simple_timing.py jan14_1142243F_norm_expr_morpho.mat")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found")
        sys.exit(1)
    
    results = run_timing_comparison(filename)
    
    print(f"\Timing comparison completed successfully!")

if __name__ == "__main__":
    main()