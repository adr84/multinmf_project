import torch
import numpy as np

from new_NMF_mps import (
    update_U_mps, update_V_mps, compute_reconstruction_error_mps,
    check_mps_memory, normalize_matrices_mps, optimize_for_m1
)


def per_view_nmf_mps(A, B, U_init, V_A_init, V_B_init, 
                    min_iter=50, max_iter=200, tol=1e-6, 
                    lambda_A=1.0, lambda_B=1.0, verbose=True):
    """
    M1/MPS-optimized Multi-view NMF
    
    Args:
        A: First view matrix (genes x cells) - torch.Tensor on MPS
        B: Second view matrix (features x cells) - torch.Tensor on MPS
        U_init: Initial consensus matrix (cells x components)
        V_A_init: Initial basis matrix for view A (genes x components)
        V_B_init: Initial basis matrix for view B (features x components)
        min_iter: Minimum iterations
        max_iter: Maximum iterations
        tol: Convergence tolerance
        lambda_A: Regularization parameter for view A
        lambda_B: Regularization parameter for view B
        verbose: Print progress
    
    Returns:
        U_final: Final consensus matrix
        V_A_final: Final basis matrix for view A
        V_B_final: Final basis matrix for view B
        final_error: Final total error
    """
    device = A.device
    eps = 1e-10
    
    # Initialize matrices and optimize for M1
    U = optimize_for_m1(U_init.clone())
    V_A = optimize_for_m1(V_A_init.clone())
    V_B = optimize_for_m1(V_B_init.clone())
    U_C = optimize_for_m1(U_init.clone())  # Consensus matrix
    
    n_A, m = A.shape
    t_B, _ = B.shape
    _, k = U.shape
    
    if verbose:
        print(f"M1 MultiNMF - A: {A.shape}, B: {B.shape}, k: {k}")
        check_mps_memory()
    
    prev_total_error = float('inf')
    
    for iteration in range(max_iter):
        # Step 1: Update V_A and V_B with fixed U and U_C
        V_A = update_V_multiview_mps(V_A, A, U, eps)
        V_B = update_V_multiview_mps(V_B, B, U, eps)
        
        # Step 2: Update U matrices with fixed V_A, V_B, and U_C
        U = update_U_multiview_mps(U, A, B, V_A, V_B, U_C, lambda_A, lambda_B, eps)
        
        # Step 3: Update consensus matrix U_C
        U_C = update_consensus_matrix_mps(U, lambda_A, lambda_B, eps)
        
        # Check convergence (less frequently to reduce CPU-GPU sync overhead)
        if iteration >= min_iter and iteration % 15 == 0:
            # Compute total error
            error_A = compute_reconstruction_error_mps(A, U, V_A)
            error_B = compute_reconstruction_error_mps(B, U, V_B)
            
            # Consensus regularization terms
            reg_A = lambda_A * torch.norm(U - U_C, p='fro').item() ** 2
            reg_B = lambda_B * torch.norm(U - U_C, p='fro').item() ** 2
            
            total_error = error_A + error_B + reg_A + reg_B
            
            if verbose and iteration % 50 == 0:
                print(f"Iteration {iteration}: Total Error = {total_error:.6f}")
                print(f"  Reconstruction A: {error_A:.6f}, B: {error_B:.6f}")
                print(f"  Regularization A: {reg_A:.6f}, B: {reg_B:.6f}")
            
            # Check convergence
            if abs(prev_total_error - total_error) < tol:
                if verbose:
                    print(f"Converged at iteration {iteration}")
                break
            
            prev_total_error = total_error
    
    # Final error computation
    final_error_A = compute_reconstruction_error_mps(A, U, V_A)
    final_error_B = compute_reconstruction_error_mps(B, U, V_B)
    final_reg_A = lambda_A * torch.norm(U - U_C, p='fro').item() ** 2
    final_reg_B = lambda_B * torch.norm(U - U_C, p='fro').item() ** 2
    final_total_error = final_error_A + final_error_B + final_reg_A + final_reg_B
    
    if verbose:
        print(f"Final total error: {final_total_error:.6f}")
        check_mps_memory()
    
    return U_C, V_A, V_B, final_total_error

def update_V_multiview_mps(V, X, U, eps=1e-10):
    """
    Update V matrix for multi-view NMF optimized for M1
    V = V * (X @ U) / (V @ (U.T @ U))
    """
    XU = torch.mm(X, U)
    UTU = torch.mm(U.T, U)
    VUTU = torch.mm(V, UTU)
    
    VUTU_safe = torch.clamp(VUTU, min=eps)
    result = V * (XU / VUTU_safe)
    
    return optimize_for_m1(result)

def update_U_multiview_mps(U, A, B, V_A, V_B, U_C, lambda_A, lambda_B, eps=1e-10):
    """
    Update U matrix for multi-view NMF with consensus regularization
    Optimized for M1's unified memory architecture
    """
    # Compute gradients from reconstruction terms
    ATV_A = torch.mm(A.T, V_A)
    V_A_TV_A = torch.mm(V_A.T, V_A)
    UV_A_TV_A = torch.mm(U, V_A_TV_A)
    
    BTV_B = torch.mm(B.T, V_B)
    V_B_TV_B = torch.mm(V_B.T, V_B)
    UV_B_TV_B = torch.mm(U, V_B_TV_B)
    
    # Numerator: reconstruction terms + consensus regularization
    numerator = ATV_A + BTV_B + (lambda_A + lambda_B) * U_C
    
    # Denominator: U terms + consensus regularization
    denominator = UV_A_TV_A + UV_B_TV_B + (lambda_A + lambda_B) * U
    
    denominator_safe = torch.clamp(denominator, min=eps)
    result = U * (numerator / denominator_safe)
    
    return optimize_for_m1(result)

def update_consensus_matrix_mps(U, lambda_A, lambda_B, eps=1e-10):
    """
    Update consensus matrix U_C for M1
    In the simplified case, consensus matrix is just the current U
    """
    return optimize_for_m1(U.clone())

def compute_multiview_error_mps(A, B, U, V_A, V_B, U_C, lambda_A, lambda_B):
    """Compute total multi-view NMF error on M1"""
    error_A = compute_reconstruction_error_mps(A, U, V_A)
    error_B = compute_reconstruction_error_mps(B, U, V_B)
    
    reg_A = lambda_A * torch.norm(U - U_C, p='fro').item() ** 2
    reg_B = lambda_B * torch.norm(U - U_C, p='fro').item() ** 2
    
    return error_A + error_B + reg_A + reg_B

def run_multiview_nmf_single_mps(A, B, k=20, max_iter=200, min_iter=50, 
                                seed=None, lambda_A=1.0, lambda_B=1.0):
    """
    Run a single instance of multi-view NMF on M1
    
    Args:
        A: First view matrix
        B: Second view matrix
        k: Number of components
        max_iter: Maximum iterations
        min_iter: Minimum iterations
        seed: Random seed
        lambda_A: Regularization for view A
        lambda_B: Regularization for view B
    
    Returns:
        U, V_A, V_B, error
    """
    device = A.device
    n_A, m = A.shape
    t_B, _ = B.shape
    
    # Initialize matrices
    if seed is not None:
        torch.manual_seed(seed)
    
    U_init = torch.rand(m, k, device=device, dtype=torch.float32)
    V_A_init = torch.rand(n_A, k, device=device, dtype=torch.float32)
    V_B_init = torch.rand(t_B, k, device=device, dtype=torch.float32)
    
    # Optimize for M1
    U_init = optimize_for_m1(U_init)
    V_A_init = optimize_for_m1(V_A_init)
    V_B_init = optimize_for_m1(V_B_init)
    
    # Run multi-view NMF
    U, V_A, V_B, error = per_view_nmf_mps(
        A, B, U_init, V_A_init, V_B_init,
        min_iter=min_iter, max_iter=max_iter,
        lambda_A=lambda_A, lambda_B=lambda_B,
        verbose=False
    )
    
    return U, V_A, V_B, error

def batch_multiview_nmf_mps(A, B, k=20, n_runs=10, max_iter=200, min_iter=50,
                           lambda_A=1.0, lambda_B=1.0):
    """
    Run multiple instances of multi-view NMF and return the best result
    Optimized for M1's memory management
    """
    best_error = float('inf')
    best_U = None
    best_V_A = None
    best_V_B = None
    
    all_errors = []
    
    print(f"Running {n_runs} multi-view NMF instances on M1...")
    
    for run in range(n_runs):
        print(f"Run {run + 1}/{n_runs}")
        
        U, V_A, V_B, error = run_multiview_nmf_single_mps(
            A, B, k=k, max_iter=max_iter, min_iter=min_iter,
            seed=run * 42, lambda_A=lambda_A, lambda_B=lambda_B
        )
        
        all_errors.append(error)
        
        if error < best_error:
            best_error = error
            best_U = U.clone()
            best_V_A = V_A.clone()
            best_V_B = V_B.clone()
        
        print(f"Run {run + 1} error: {error:.6f}")
        
        # M1 memory management after each run
        if hasattr(torch.backends.mps, 'empty_cache'):
            torch.backends.mps.empty_cache()
    
    print(f"Best error: {best_error:.6f}")
    
    return best_U, best_V_A, best_V_B, all_errors

def memory_efficient_multiview_mps(A, B, k=20, max_iter=200, chunk_size=2000):
    """
    Memory-efficient multi-view NMF for large datasets on M1
    Takes advantage of M1's unified memory architecture
    """
    device = A.device
    n_A, m = A.shape
    t_B, _ = B.shape
    
    # Initialize matrices
    U = torch.rand(m, k, device=device, dtype=torch.float32)
    V_A = torch.rand(n_A, k, device=device, dtype=torch.float32)
    V_B = torch.rand(t_B, k, device=device, dtype=torch.float32)
    U_C = U.clone()
    
    eps = 1e-10
    lambda_A = lambda_B = 1.0
    
    print(f"Memory-efficient M1 MultiNMF: A{A.shape}, B{B.shape}, chunk_size={chunk_size}")
    
    for iteration in range(max_iter):
        # Process A in chunks if needed
        if n_A > chunk_size:
            for i in range(0, n_A, chunk_size):
                end_idx = min(i + chunk_size, n_A)
                A_chunk = A[i:end_idx, :]
                V_A_chunk = V_A[i:end_idx, :]
                
                V_A_chunk_updated = update_V_multiview_mps(V_A_chunk, A_chunk, U, eps)
                V_A[i:end_idx, :] = V_A_chunk_updated
        else:
            V_A = update_V_multiview_mps(V_A, A, U, eps)
        
        # Process B in chunks if needed
        if t_B > chunk_size:
            for i in range(0, t_B, chunk_size):
                end_idx = min(i + chunk_size, t_B)
                B_chunk = B[i:end_idx, :]
                V_B_chunk = V_B[i:end_idx, :]
                
                V_B_chunk_updated = update_V_multiview_mps(V_B_chunk, B_chunk, U, eps)
                V_B[i:end_idx, :] = V_B_chunk_updated
        else:
            V_B = update_V_multiview_mps(V_B, B, U, eps)
        
        # Update U (usually smaller matrix)
        U = update_U_multiview_mps(U, A, B, V_A, V_B, U_C, lambda_A, lambda_B, eps)
        U_C = update_consensus_matrix_mps(U, lambda_A, lambda_B, eps)
        
        # Check progress less frequently for large matrices
        if iteration % 25 == 0:
            error = compute_multiview_error_mps(A, B, U, V_A, V_B, U_C, lambda_A, lambda_B)
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Error: {error:.6f}")
                check_mps_memory()
    
    final_error = compute_multiview_error_mps(A, B, U, V_A, V_B, U_C, lambda_A, lambda_B)
    return U_C, V_A, V_B, final_error

def benchmark_multiview_mps():
    """Benchmark multi-view NMF performance on M1"""
    if not torch.backends.mps.is_available():
        print("MPS not available for benchmarking")
        return
    
    device = torch.device("mps")
    sizes = [(1000, 500), (2000, 800), (4000, 1000)]
    k = 20
    
    print("M1 Multi-view NMF Benchmark:")
    print("-" * 32)
    
    for n_genes, n_features in sizes:
        n_cells = 500
        
        # Create test matrices
        A = torch.rand(n_genes, n_cells, device=device, dtype=torch.float32)
        B = torch.rand(n_features, n_cells, device=device, dtype=torch.float32)
        
        print(f"Testing A{A.shape} + B{B.shape}:")
        
        import time
        start_time = time.time()
        
        # Run single iteration for timing
        U, V_A, V_B, error = run_multiview_nmf_single_mps(
            A, B, k=k, max_iter=50, min_iter=10
        )
        
        elapsed = time.time() - start_time
        print(f"  Time: {elapsed:.2f}s, Error: {error:.6f}")
        
        # Clean up
        del A, B, U, V_A, V_B
        if hasattr(torch.backends.mps, 'empty_cache'):
            torch.backends.mps.empty_cache()
