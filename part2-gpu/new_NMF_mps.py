import torch
import numpy as np

def get_device():
    """Get the appropriate device for computation"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def nmf_mps(X, k, max_iter=200, tol=1e-6, device=None):
    """
    M1/MPS-optimized Non-negative Matrix Factorization
    
    Args:
        X: Input matrix (torch.Tensor) on MPS/GPU
        k: Number of components
        max_iter: Maximum iterations
        tol: Tolerance for convergence
        device: PyTorch device (auto-detected if None)
    
    Returns:
        U: Coefficient matrix (samples x components)
        V: Basis matrix (features x components)
        error: Final reconstruction error
    """
    if device is None:
        device = X.device
    
    n, m = X.shape
    
    # Initialize U and V with random positive values
    # Use float32 for optimal M1 performance
    U = torch.rand(m, k, device=device, dtype=torch.float32)
    V = torch.rand(n, k, device=device, dtype=torch.float32)
    
    # Small epsilon to avoid division by zero
    eps = 1e-10
    
    prev_error = float('inf')
    
    for iteration in range(max_iter):
        # Update V using multiplicative update rule
        # V = V * (X @ U) / (V @ (U.T @ U))
        XU = torch.mm(X, U)
        UTU = torch.mm(U.T, U)
        VUTU = torch.mm(V, UTU)
        
        # Avoid division by zero with clamping (M1-optimized)
        VUTU_safe = torch.clamp(VUTU, min=eps)
        V = V * (XU / VUTU_safe)
        
        # Update U using multiplicative update rule
        # U = U * (X.T @ V) / (U @ (V.T @ V))
        XTV = torch.mm(X.T, V)
        VTV = torch.mm(V.T, V)
        UVTV = torch.mm(U, VTV)
        
        # Avoid division by zero
        UVTV_safe = torch.clamp(UVTV, min=eps)
        U = U * (XTV / UVTV_safe)
        
        # Check convergence every 10 iterations to reduce CPU-GPU sync
        if iteration % 10 == 0:
            # Compute reconstruction error
            reconstruction = torch.mm(V, U.T)
            error = torch.norm(X - reconstruction, p='fro').item() ** 2
            
            if abs(prev_error - error) < tol:
                print(f"Converged at iteration {iteration} with error {error:.6f}")
                break
            
            prev_error = error
    
    # Final error computation
    reconstruction = torch.mm(V, U.T)
    final_error = torch.norm(X - reconstruction, p='fro').item() ** 2
    
    return U, V, final_error

def matrix_multiply_mps(A, B):
    """MPS-optimized matrix multiplication with error checking"""
    try:
        return torch.mm(A, B)
    except RuntimeError as e:
        print(f"Matrix multiplication error: {e}")
        print(f"A shape: {A.shape}, B shape: {B.shape}")
        raise

def element_wise_multiply_mps(A, B):
    """MPS-optimized element-wise multiplication"""
    return A * B

def element_wise_divide_mps(A, B, eps=1e-10):
    """MPS-optimized element-wise division with numerical stability"""
    B_safe = torch.clamp(B, min=eps)
    return A / B_safe

def compute_frobenius_norm_mps(X):
    """Compute Frobenius norm on MPS device"""
    return torch.norm(X, p='fro')

def update_U_mps(U, X, V, eps=1e-10):
    """
    MPS-optimized update for U matrix
    U = U * (X.T @ V) / (U @ (V.T @ V))
    """
    XTV = torch.mm(X.T, V)
    VTV = torch.mm(V.T, V)
    UVTV = torch.mm(U, VTV)
    
    # Numerical stability
    UVTV_safe = torch.clamp(UVTV, min=eps)
    
    return U * (XTV / UVTV_safe)

def update_V_mps(V, X, U, eps=1e-10):
    """
    MPS-optimized update for V matrix
    V = V * (X @ U) / (V @ (U.T @ U))
    """
    XU = torch.mm(X, U)
    UTU = torch.mm(U.T, U)
    VUTU = torch.mm(V, UTU)
    
    # Numerical stability
    VUTU_safe = torch.clamp(VUTU, min=eps)
    
    return V * (XU / VUTU_safe)

def compute_reconstruction_error_mps(X, U, V):
    """Compute reconstruction error ||X - VU^T||_F^2 on MPS"""
    reconstruction = torch.mm(V, U.T)
    error = torch.norm(X - reconstruction, p='fro').item() ** 2
    return error

def initialize_matrices_mps(n, m, k, device, seed=None):
    """Initialize U and V matrices on MPS device"""
    if seed is not None:
        torch.manual_seed(seed)
    
    # Use float32 for better M1 performance
    U = torch.rand(m, k, device=device, dtype=torch.float32)
    V = torch.rand(n, k, device=device, dtype=torch.float32)
    
    return U, V

def normalize_matrices_mps(U, V):
    """Normalize U and V matrices on MPS device"""
    # Normalize columns of V
    V_norm = torch.norm(V, dim=0, keepdim=True)
    V_norm_safe = torch.clamp(V_norm, min=1e-10)
    V_normalized = V / V_norm_safe
    
    # Scale U accordingly
    U_scaled = U * V_norm.T
    
    return U_scaled, V_normalized

def check_mps_memory():
    """Check MPS memory usage (M1 unified memory)"""
    if torch.backends.mps.is_available():
        # M1 uses unified memory, so we check system memory
        import psutil
        memory = psutil.virtual_memory()
        total_gb = memory.total / 1024**3
        available_gb = memory.available / 1024**3
        used_gb = memory.used / 1024**3
        
        print(f"M1 Unified Memory - Total: {total_gb:.1f} GB, Used: {used_gb:.1f} GB, Available: {available_gb:.1f} GB")
        return used_gb, available_gb
    return 0, 0

def optimize_for_m1(tensor):
    """Apply M1-specific optimizations to tensors"""
    # Ensure tensor is contiguous for better performance
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    
    # M1 works best with float32
    if tensor.dtype != torch.float32:
        tensor = tensor.to(torch.float32)
    
    return tensor

def batch_matrix_multiply_mps(A_list, B_list):
    """
    Batch matrix multiplication optimized for M1
    Useful for processing multiple matrix pairs efficiently
    """
    if len(A_list) != len(B_list):
        raise ValueError("A_list and B_list must have same length")
    
    results = []
    for A, B in zip(A_list, B_list):
        # Optimize tensors for M1
        A = optimize_for_m1(A)
        B = optimize_for_m1(B)
        
        # Perform multiplication
        result = torch.mm(A, B)
        results.append(result)
    
    return results

def memory_efficient_nmf_mps(X, k, max_iter=200, chunk_size=1000):
    """
    Memory-efficient NMF for large matrices on M1
    Uses chunking to handle large datasets
    """
    device = X.device
    n, m = X.shape
    
    # Initialize matrices
    U = torch.rand(m, k, device=device, dtype=torch.float32)
    V = torch.rand(n, k, device=device, dtype=torch.float32)
    
    eps = 1e-10
    
    for iteration in range(max_iter):
        # Process in chunks if matrix is large
        if n > chunk_size:
            # Chunked V update
            for i in range(0, n, chunk_size):
                end_idx = min(i + chunk_size, n)
                X_chunk = X[i:end_idx, :]
                V_chunk = V[i:end_idx, :]
                
                XU = torch.mm(X_chunk, U)
                UTU = torch.mm(U.T, U)
                VUTU = torch.mm(V_chunk, UTU)
                VUTU_safe = torch.clamp(VUTU, min=eps)
                
                V[i:end_idx, :] = V_chunk * (XU / VUTU_safe)
        else:
            # Standard V update
            V = update_V_mps(V, X, U, eps)
        
        # U update (usually smaller, no chunking needed)
        U = update_U_mps(U, X, V, eps)
        
        # Check convergence less frequently for large matrices
        if iteration % 20 == 0:
            error = compute_reconstruction_error_mps(X, U, V)
            if iteration % 100 == 0:
                print(f"Iteration {iteration}, Error: {error:.6f}")
    
    final_error = compute_reconstruction_error_mps(X, U, V)
    return U, V, final_error

def benchmark_mps_operations():
    """Benchmark basic MPS operations for performance testing"""
    if not torch.backends.mps.is_available():
        print("MPS not available for benchmarking")
        return
    
    import time
    
    device = torch.device("mps")
    sizes = [500, 1000, 2000, 4000]
    
    print("M1 MPS Performance Benchmark:")
    print("-" * 35)
    
    for size in sizes:
        # Create test matrices
        A = torch.randn(size, size, device=device, dtype=torch.float32)
        B = torch.randn(size, size, device=device, dtype=torch.float32)
        
        # Matrix multiplication benchmark
        start_time = time.time()
        C = torch.mm(A, B)
        
        # Synchronize to get accurate timing
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'synchronize'):
            torch.mps.synchronize()
        
        elapsed = time.time() - start_time
        
        # Calculate performance metrics
        operations = 2 * size**3  # Multiply-add operations
        gflops = operations / elapsed / 1e9
        
        print(f"Size {size:4d}x{size:<4d}: {elapsed:.3f}s, {gflops:.1f} GFLOPS")
        
        del A, B, C
        if hasattr(torch.backends.mps, 'empty_cache'):
            torch.backends.mps.empty_cache()

def get_mps_info():
    """Get M1/MPS system information"""
    info = {
        'mps_available': torch.backends.mps.is_available(),
        'mps_built': torch.backends.mps.is_built() if hasattr(torch.backends.mps, 'is_built') else True,
        'pytorch_version': torch.__version__
    }
    
    if info['mps_available']:
        try:
            test_tensor = torch.randn(10, 10, device='mps')
            result = torch.mm(test_tensor, test_tensor.T)
            info['mps_functional'] = True
            del test_tensor, result
        except:
            info['mps_functional'] = False
    else:
        info['mps_functional'] = False
    
    return info
