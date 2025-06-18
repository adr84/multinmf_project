from mpi4py import MPI
import numpy as np
import time
import sys
import os
from scipy.io import loadmat

def load_data_on_master(filename):
    """Load data only on master process"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:  
        print(f"Loading data from {filename}...")
        data = loadmat(filename)
        
        if 'expr' in data and 'morpho' in data:
            A = data['expr'].astype(np.float32)
            B = data['morpho'].astype(np.float32)
        elif 'A' in data and 'B' in data:
            A = data['A'].astype(np.float32)
            B = data['B'].astype(np.float32)
        else:
            keys = [k for k in data.keys() if not k.startswith('__')]
            A = data[keys[0]].astype(np.float32)
            B = data[keys[1]].astype(np.float32)
        
        min_cols = min(A.shape[1], B.shape[1])
        A = A[:, :min_cols]
        B = B[:, :min_cols]
        
        print(f"Master: Matrix A: {A.shape}, Matrix B: {B.shape}")
        return A, B
    else:
        return None, None

def broadcast_data(A, B):
    """Broadcast data from master to all processes"""
    comm = MPI.COMM_WORLD
    A = comm.bcast(A, root=0)
    B = comm.bcast(B, root=0)
    return A, B

def fixed_multinmf_single_run(A, B, k=20, max_iter=200, seed=42):
    """Fixed single MultiNMF run - simplified and robust"""
    np.random.seed(seed)
    
    n_A, m = A.shape
    t_B, _ = B.shape
    
    U = np.random.rand(m, k).astype(np.float32) + 0.1
    V_A = np.random.rand(n_A, k).astype(np.float32) + 0.1
    V_B = np.random.rand(t_B, k).astype(np.float32) + 0.1
    
    eps = 1e-10
    
    for iteration in range(max_iter):
        numerator_VA = A @ U
        denominator_VA = V_A @ (U.T @ U) + eps
        V_A = V_A * (numerator_VA / denominator_VA)
        
        numerator_VB = B @ U
        denominator_VB = V_B @ (U.T @ U) + eps
        V_B = V_B * (numerator_VB / denominator_VB)
        
        numerator_U = A.T @ V_A + B.T @ V_B
        denominator_U = U @ (V_A.T @ V_A) + U @ (V_B.T @ V_B) + eps
        U = U * (numerator_U / denominator_U)
    
    error_A = np.linalg.norm(A - V_A @ U.T, 'fro') ** 2
    error_B = np.linalg.norm(B - V_B @ U.T, 'fro') ** 2
    
    return U, V_A, V_B, error_A + error_B

def optimized_load_balancing(total_runs, size):
    """Better load balancing"""
    runs_per_process = total_runs // size
    extra_runs = total_runs % size
    
    work_distribution = []
    current_run = 0
    
    for rank in range(size):
        if rank < extra_runs:
            my_runs = runs_per_process + 1
        else:
            my_runs = runs_per_process
        
        run_ids = list(range(current_run, current_run + my_runs))
        work_distribution.append(run_ids)
        current_run += my_runs
    
    return work_distribution

def mpi_multinmf_parallel_runs(A, B, total_runs=10, k=20, max_iter=200):
    """Fixed MPI MultiNMF with better load balancing"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    work_distribution = optimized_load_balancing(total_runs, size)
    my_runs = work_distribution[rank]
    
    if rank == 0:
        print(f"Distributing {total_runs} runs across {size} processes")
        print(f"Load balancing: {[len(work) for work in work_distribution]}")
    
    local_results = []
    local_errors = []
    
    comm.Barrier()
    local_start_time = time.time()
    
    for run_id in my_runs:
        seed = run_id * 42  
        
        if rank == 0 and len(my_runs) > 0:
            print(f"Process {rank}: Starting run {run_id + 1}/{total_runs}")
        
        U, V_A, V_B, error = fixed_multinmf_single_run(A, B, k, max_iter, seed)
        
        local_results.append({
            'U': U,
            'V_A': V_A, 
            'V_B': V_B,
            'error': error,
            'run_id': run_id
        })
        local_errors.append(error)
        
        print(f"Process {rank}: Run {run_id + 1} completed, error: {error:.6f}")
    
    local_time = time.time() - local_start_time
    
    all_results = comm.gather(local_results, root=0)
    all_errors = comm.gather(local_errors, root=0)
    all_times = comm.gather(local_time, root=0)
    
    if rank == 0:
        final_results = []
        final_errors = []
        
        for process_results in all_results:
            final_results.extend(process_results)
            
        for process_errors in all_errors:
            final_errors.extend(process_errors)
        
        final_results.sort(key=lambda x: x['run_id'])
        
        best_idx = np.argmin([r['error'] for r in final_results])
        best_result = final_results[best_idx]
        
        max_time = max(all_times)
        avg_time = np.mean(all_times)
        load_balance_efficiency = (avg_time / max_time) * 100
        
        print(f"\nMPI MultiNMF completed!")
        print(f"Best result: Run {best_result['run_id'] + 1}, Error: {best_result['error']:.6f}")
        print(f"Average error: {np.mean(final_errors):.6f}")
        print(f"Error std: {np.std(final_errors):.6f}")
        print(f"Max process time: {max_time:.2f} seconds")
        print(f"Avg process time: {avg_time:.2f} seconds")
        print(f"Load balance efficiency: {load_balance_efficiency:.1f}%")
        
        return best_result, final_errors, max_time
    else:
        return None, None, None

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if len(sys.argv) != 2:
        if rank == 0:
            print("Usage: mpirun -np N python mpi_multinmf_fixed.py <input_file.mat>")
            print("Example: mpirun -np 4 python mpi_multinmf_fixed.py jan14_1142243F_norm_expr_morpho.mat")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    if rank == 0:
        print(f"Fixed MPI MultiNMF starting with {size} processes")
        if not os.path.exists(filename):
            print(f"Error: File '{filename}' not found")
            comm.Abort(1)
        start_time = time.time()
    
    A, B = load_data_on_master(filename)
    A, B = broadcast_data(A, B)
    
    if rank != 0:
        print(f"Process {rank}: Received data - A: {A.shape}, B: {B.shape}")
    
    result, errors, compute_time = mpi_multinmf_parallel_runs(A, B, total_runs=10)
    
    if rank == 0:
        total_time = time.time() - start_time
        print(f"\nFixed MPI Performance Summary:")
        print(f"Total MPI time: {total_time:.2f} seconds")
        print(f"Pure computation time: {compute_time:.2f} seconds")
        print(f"Communication overhead: {total_time - compute_time:.2f} seconds")
        print(f"Computation efficiency: {(compute_time/total_time)*100:.1f}%")
        print(f"Average time per run: {compute_time/10:.2f} seconds")
        print(f"MPI processes used: {size}")
        
        if result:
            np.savez('mpi_fixed_results.npz',
                     U_final=result['U'],
                     V_A_final=result['V_A'],
                     V_B_final=result['V_B'],
                     errors=errors,
                     mpi_processes=size,
                     total_time=total_time,
                     compute_time=compute_time,
                     runs_distributed=10)
            print(f"Results saved to mpi_fixed_results.npz")

if __name__ == "__main__":
    main()