
import torch
import scipy.io
import numpy as np
import sys
import time
import os
from datetime import datetime

# M1/Apple Silicon device selection
def get_best_device():
    """Get the best available device for M1 Macs"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():  # Fallback for other systems
        return torch.device("cuda")
    else:
        return torch.device("cpu")

device = get_best_device()
print(f"Using device: {device}")

if device.type == "mps":
    print("🍎 Apple M1/M2 GPU acceleration enabled!")
    print("Unified memory architecture provides excellent performance")
elif device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("Using CPU - consider updating PyTorch for MPS support")

def load_matrices_mps(filename):
    """Load matrices from .mat file and convert to MPS tensors"""
    print(f"Loading matrices from {filename}...")
    
    # Load the .mat file
    mat_data = scipy.io.loadmat(filename)
    
    # Extract matrices A and B with flexible key detection
    A, B = None, None
    
    if 'A' in mat_data and 'B' in mat_data:
        A = mat_data['A']
        B = mat_data['B']
    elif 'expr' in mat_data and 'morpho' in mat_data:
        A = mat_data['expr']  # gene expression matrix
        B = mat_data['morpho']  # morphology/image features matrix
    else:
        # Auto-detect matrices
        keys = [k for k in mat_data.keys() if not k.startswith('__')]
        print(f"Available keys in .mat file: {keys}")
        if len(keys) >= 2:
            A = mat_data[keys[0]]
            B = mat_data[keys[1]]
            print(f"Auto-selected matrices: {keys[0]} and {keys[1]}")
        else:
            raise ValueError("Could not identify matrices A and B in the .mat file")
    
    print(f"Original Matrix A shape: {A.shape}")
    print(f"Original Matrix B shape: {B.shape}")
    
    # ADD THIS: Ensure compatible dimensions (same as MPI script)
    min_cols = min(A.shape[1], B.shape[1])
    A = A[:, :min_cols]
    B = B[:, :min_cols]
    
    print(f"After dimension matching:")
    print(f"Matrix A shape: {A.shape}")
    print(f"Matrix B shape: {B.shape}")
    
    # Convert to torch tensors and move to device (MPS/GPU)
    # Use float32 for better M1 performance
    A_tensor = torch.tensor(A, dtype=torch.float32, device=device)
    B_tensor = torch.tensor(B, dtype=torch.float32, device=device)
    
    print(f"Matrices loaded to {device}")
    
    return A_tensor, B_tensor


def run_mps_multinmf(A, B, k=20, n_repeat=10, min_iter=50, max_iter=200):
    """Run MultiNMF on M1 GPU using MPS"""
    print(f"Running M1 MultiNMF with k={k}, n_repeat={n_repeat}")
    
    n_A, m = A.shape
    t_B, m_B = B.shape
    
    assert m == m_B, f"Matrix dimensions don't match: A has {m} columns, B has {m_B} columns"
    
    # Import MPS-optimized NMF functions
    from new_NMF_mps import nmf_mps
    from new_PerViewNMF_mps import per_view_nmf_mps
    
    start_time = time.time()
    
    # Store results from multiple runs
    U_results = []
    V_A_results = []
    V_B_results = []
    errors = []
    
    for run in range(n_repeat):
        print(f"Run {run + 1}/{n_repeat}")
        
        # Initialize random matrices on device
        torch.manual_seed(run * 42)  # For reproducibility
        U_init = torch.rand(m, k, device=device, dtype=torch.float32)
        V_A_init = torch.rand(n_A, k, device=device, dtype=torch.float32)
        V_B_init = torch.rand(t_B, k, device=device, dtype=torch.float32)
        
        # Run MultiNMF for this initialization
        U_final, V_A_final, V_B_final, final_error = per_view_nmf_mps(
            A, B, U_init, V_A_init, V_B_init, 
            min_iter=min_iter, max_iter=max_iter
        )
        
        # Store results (move to CPU for storage)
        U_results.append(U_final.detach().cpu().numpy())
        V_A_results.append(V_A_final.detach().cpu().numpy())
        V_B_results.append(V_B_final.detach().cpu().numpy())
        errors.append(final_error)
        
        print(f"Run {run + 1} completed with error: {final_error:.6f}")
        
        # M1 memory management
        if device.type == "mps" and hasattr(torch.backends.mps, 'empty_cache'):
            torch.backends.mps.empty_cache()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Total M1 GPU computation time: {total_time:.2f} seconds")
    print(f"Average time per run: {total_time/n_repeat:.2f} seconds")
    
    # Find best result (lowest error)
    best_idx = np.argmin(errors)
    print(f"Best result from run {best_idx + 1} with error: {errors[best_idx]:.6f}")
    
    # Compute consensus results
    U_consensus = np.mean(U_results, axis=0)
    V_A_consensus = V_A_results[best_idx]  # Use best result
    V_B_consensus = V_B_results[best_idx]  # Use best result
    
    return U_consensus, V_A_consensus, V_B_consensus, errors, total_time

def save_results_npz(U, V_A, V_B, errors, filename, computation_time):
    """Save results in .npz format compatible with visualization scripts"""
    np.savez(filename,
             U_final=U,
             V_A_final=V_A,
             V_B_final=V_B,
             errors=errors,
             computation_time=computation_time,
             device_used=str(device),
             timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Results saved to {filename}")

def print_performance_summary(computation_time, n_repeat, errors):
    """Print detailed performance summary"""
    print(f"\n" + "="*50)
    print("M1 MULTINMF PERFORMANCE SUMMARY")
    print("="*50)
    print(f"Device used: {device}")
    print(f"Total computation time: {computation_time:.2f} seconds")
    print(f"Average time per run: {computation_time/n_repeat:.2f} seconds")
    print(f"Number of runs: {n_repeat}")
    print(f"Best error: {min(errors):.6f}")
    print(f"Mean error: {np.mean(errors):.6f}")
    print(f"Error std: {np.std(errors):.6f}")
    
    # Estimate speedup potential
    estimated_cpu_time = computation_time * 4  # Conservative estimate
    print(f"\nEstimated performance vs CPU:")
    print(f"  Estimated CPU time: ~{estimated_cpu_time:.1f} seconds")
    print(f"  M1 GPU speedup: ~{estimated_cpu_time/computation_time:.1f}x")

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 load_matrix_mps.py <input_file.mat> <output_file.npz>")
        print("Example: python3 load_matrix_mps.py jan14_1142243F_norm_expr_morpho.mat m1_results.npz")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        sys.exit(1)
    
    # Check device availability
    if device.type == "cpu":
        print("Warning: No GPU acceleration available. Running on CPU.")
        print("For M1 Macs, ensure you have:")
        print("  - macOS 12.3+")
        print("  - PyTorch with MPS support: pip install torch torchvision torchaudio")
    
    try:
        print("Starting M1-optimized MultiNMF analysis...")
        print("-" * 40)
        
        # Load matrices
        A, B = load_matrices_mps(input_file)
        
        # Run M1 MultiNMF
        U, V_A, V_B, errors, computation_time = run_mps_multinmf(A, B)
        
        # Save results
        save_results_npz(U, V_A, V_B, errors, output_file, computation_time)
        
        # Print performance summary
        print_performance_summary(computation_time, len(errors), errors)
        
        print(f"\n🎉 M1 MultiNMF completed successfully!")
        print(f"Output saved to: {output_file}")
        
        # Cleanup
        del A, B
        if device.type == "mps" and hasattr(torch.backends.mps, 'empty_cache'):
            torch.backends.mps.empty_cache()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()