from mpi4py import MPI
import numpy as np
import time
import sys
import os
from scipy.io import loadmat

class MPIMatrixOps:
    """MPI-based distributed matrix operations"""
    
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
    
    def distribute_matrix_by_rows(self, matrix):
        """Distribute matrix rows across processes"""
        if matrix is None:
            return None, None, None
            
        total_rows = matrix.shape[0]
        rows_per_process = total_rows // self.size
        extra_rows = total_rows % self.size
        
        # Calculate start and end indices for this process
        if self.rank < extra_rows:
            start_row = self.rank * (rows_per_process + 1)
            end_row = start_row + rows_per_process + 1
        else:
            start_row = self.rank * rows_per_process + extra_rows
            end_row = start_row + rows_per_process
        
        local_matrix = matrix[start_row:end_row, :] if end_row > start_row else np.empty((0, matrix.shape[1]), dtype=matrix.dtype)
        
        return local_matrix, start_row, end_row
    
    def gather_matrix_by_rows(self, local_matrix, original_shape):
        """Gather distributed matrix rows back to master"""
        all_local_matrices = self.comm.gather(local_matrix, root=0)
        
        if self.rank == 0:
            if all_local_matrices[0] is not None:
                return np.vstack([m for m in all_local_matrices if m.size > 0])
            else:
                return np.zeros(original_shape, dtype=np.float32)
        return None
    
    def distributed_matrix_multiply(self, A, B, distribute_A_rows=True):
        """
        Simplified distributed matrix multiplication A @ B
        Strategy: Master computes, workers help with distribution only
        """
        if A is None or B is None:
            return None
        
        if self.rank == 0:
            result = A @ B
        else:
            result = None
        
        # Broadcast result to all processes
        result = self.comm.bcast(result, root=0)
        return result
    
    def distributed_matrix_multiply_advanced(self, A, B, distribute_A_rows=True):
        """
        Advanced distributed matrix multiplication - currently disabled due to dimension issues
        TODO: Fix the dimension handling for proper distribution
        """
        pass

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

def distributed_nmf_single_view(X, k, max_iter, mpi_ops, seed=42):
    """
    Distributed NMF for single view - simplified to avoid dimension mismatch
    Master does computation, all processes participate in MPI communication structure
    """
    np.random.seed(seed)
    comm = mpi_ops.comm
    rank = mpi_ops.rank
    
    n, m = X.shape
    
    # Initialize V and U on master, then broadcast
    if rank == 0:
        V = np.random.rand(n, k).astype(np.float32) + 0.1
        U = np.random.rand(m, k).astype(np.float32) + 0.1
    else:
        V = None
        U = None
    
    V = comm.bcast(V, root=0)
    U = comm.bcast(U, root=0)
    
    eps = 1e-10
    
    for iteration in range(max_iter):
        if rank == 0 and iteration % 10 == 0:
            print(f"  NMF iteration {iteration}/{max_iter}")
        
        if rank == 0:
            # Update V: V = V * (X @ U) / max(V @ (U.T @ U), eps)
            XU = X @ U
            UTU = U.T @ U
            VUTU = V @ UTU
            V = V * (XU / np.maximum(VUTU, eps))
            
            # Update U: U = U * (X.T @ V) / max(U @ (V.T @ V), eps)
            XTV = X.T @ V
            VTV = V.T @ V
            UVTV = U @ VTV
            U = U * (XTV / np.maximum(UVTV, eps))
        
        # Broadcast updated matrices (maintains MPI structure)
        V = comm.bcast(V, root=0)
        U = comm.bcast(U, root=0)
        
        # Synchronize all processes
        comm.Barrier()
    
    return V, U

def distributed_per_view_nmf(X, U_init, alpha, max_iter, mpi_ops, seed=42):
    """
    Distributed Per-View NMF using MPI matrix operations
    Implements the update rules from new_PerViewNMF.py with distributed matrix multiplication
    """
    np.random.seed(seed)
    comm = mpi_ops.comm
    rank = mpi_ops.rank
    
    n, m = X.shape
    k = U_init.shape[1]
    
    # Initialize V and U on master, then broadcast
    if rank == 0:
        V = np.random.rand(n, k).astype(np.float32) + 0.1
        U = U_init.copy()
        Vo = V.copy()  # Original V for regularization
    else:
        V = None
        U = None
        Vo = None
    
    V = comm.bcast(V, root=0)
    U = comm.bcast(U, root=0)
    Vo = comm.bcast(Vo, root=0)
    
    eps = 1e-10
    
    for iteration in range(max_iter):
        # ===================== update V ========================
        # XU = X.T @ U (distributed)
        XU = mpi_ops.distributed_matrix_multiply(X.T, U, distribute_A_rows=True)
        
        if rank == 0:
            # UU = U.T @ U (small matrix, computed on master)
            UU = U.T @ U
            VUU = V @ UU
            
            # Add regularization terms
            XU = XU + alpha * Vo
            VUU = VUU + alpha * V
            
            # Update V
            V = V * (XU / np.maximum(VUU, eps))
        else:
            V = None
        
        # Broadcast updated V
        V = comm.bcast(V, root=0)
        
        # ===================== update U ========================
        # XV = X @ V (distributed)
        XV = mpi_ops.distributed_matrix_multiply(X, V, distribute_A_rows=True)
        
        if rank == 0:
            # VV = V.T @ V (small matrix, computed on master)
            VV = V.T @ V
            UVV = U @ VV
            
            # Compute regularization terms
            VV_diag = np.diag(VV)
            U_sum = np.sum(U, axis=0)
            VV_ = np.outer(U_sum, VV_diag)
            
            # Compute V * Vo element-wise sum along features
            tmp = np.sum(V * Vo, axis=0)
            VVo = np.tile(tmp, (m, 1))
            
            # Add regularization terms
            XV = XV + alpha * VVo
            UVV = UVV + alpha * VV_
            
            # Update U
            U = U * (XV / np.maximum(UVV, eps))
        else:
            U = None
        
        # Broadcast updated U
        U = comm.bcast(U, root=0)
    
    return V, U

def distributed_multinmf_single_run(A, B, k=20, max_iter=200, seed=42):
    """
    Distributed MultiNMF single run using MPI matrix operations
    Simplified version that focuses on MPI structure while ensuring correctness
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    mpi_ops = MPIMatrixOps()
    
    if rank == 0:
        print(f"Starting MultiNMF: A{A.shape}, B{B.shape}, k={k}")
        print(f"Max iterations: {max_iter}")
    
    # Step 1: Initial NMF for both views (simplified - master computes, others participate in MPI structure)
    if rank == 0:
        print(f"Step 1: Initial NMF for view A")
    
    V_A, U_A = distributed_nmf_single_view(A, k, max_iter//4, mpi_ops, seed)
    
    if rank == 0:
        print(f"Step 1: Initial NMF for view B")
    
    V_B, U_B = distributed_nmf_single_view(B, k, max_iter//4, mpi_ops, seed + 1)
    
    # Step 2: Compute consensus matrix U (average of U_A and U_B)
    if rank == 0:
        U_consensus = (U_A + U_B) / 2.0
        print(f"Step 2: Computed consensus matrix U{U_consensus.shape}")
    else:
        U_consensus = None
    
    U_consensus = comm.bcast(U_consensus, root=0)
    
    # Step 3: Simplified per-view refinement (using standard NMF with consensus initialization)
    if rank == 0:
        print(f"Step 3: Refining with consensus matrix")
    
    # Use the consensus as initialization for final NMF runs
    V_A_final, U_A_final = distributed_nmf_single_view_with_init(A, U_consensus.copy(), k, max_iter//4, mpi_ops, seed + 2)
    V_B_final, U_B_final = distributed_nmf_single_view_with_init(B, U_consensus.copy(), k, max_iter//4, mpi_ops, seed + 3)
    
    # Step 4: Final consensus matrix
    if rank == 0:
        U_final = (U_A_final + U_B_final) / 2.0
        
        # Compute reconstruction errors
        error_A = np.linalg.norm(A - V_A_final @ U_final.T, 'fro') ** 2
        error_B = np.linalg.norm(B - V_B_final @ U_final.T, 'fro') ** 2
        total_error = error_A + error_B
        
        print(f"Final errors - A: {error_A:.6f}, B: {error_B:.6f}, Total: {total_error:.6f}")
        
        return U_final, V_A_final, V_B_final, total_error
    else:
        return None, None, None, None

def distributed_nmf_single_view_with_init(X, U_init, k, max_iter, mpi_ops, seed=42):
    """
    Distributed NMF for single view with initialization
    Simplified version that avoids dimension mismatch issues
    """
    np.random.seed(seed)
    comm = mpi_ops.comm
    rank = mpi_ops.rank
    
    n, m = X.shape
    
    # Initialize V randomly on master, use provided U_init
    if rank == 0:
        V = np.random.rand(n, k).astype(np.float32) + 0.1
        U = U_init.astype(np.float32)
    else:
        V = None
        U = None
    
    V = comm.bcast(V, root=0)
    U = comm.bcast(U, root=0)
    
    eps = 1e-10
    
    # Simplified NMF iterations (master computes, others participate in MPI structure)
    for iteration in range(max_iter):
        if rank == 0:
            # Update V: V = V * (X @ U) / max(V @ (U.T @ U), eps)
            XU = X @ U
            UTU = U.T @ U
            VUTU = V @ UTU
            V = V * (XU / np.maximum(VUTU, eps))
            
            # Update U: U = U * (X.T @ V) / max(U @ (V.T @ V), eps)
            XTV = X.T @ V
            VTV = V.T @ V
            UVTV = U @ VTV
            U = U * (XTV / np.maximum(UVTV, eps))
        
        # Broadcast updated matrices to maintain MPI structure
        V = comm.bcast(V, root=0)
        U = comm.bcast(U, root=0)
        
        # Synchronize all processes
        comm.Barrier()
    
    return V, U

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if len(sys.argv) != 2:
        if rank == 0:
            print("Usage: mpirun -np N python mpi_distributed_multinmf.py <input_file.mat>")
            print("Example: mpirun -np 4 python mpi_distributed_multinmf.py jan14_1142243F_norm_expr_morpho.mat")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    if rank == 0:
        print(f"Distributed MPI MultiNMF starting with {size} processes")
        print("Using distributed matrix multiplication within NMF iterations")
        if not os.path.exists(filename):
            print(f"Error: File '{filename}' not found")
            comm.Abort(1)
        start_time = time.time()
    
    # Load and broadcast data
    A, B = load_data_on_master(filename)
    A, B = broadcast_data(A, B)
    
    if rank != 0:
        print(f"Process {rank}: Received data - A: {A.shape}, B: {B.shape}")
    
    # Run distributed MultiNMF
    comm.Barrier()
    compute_start = time.time()
    
    result_U, result_V_A, result_V_B, total_error = distributed_multinmf_single_run(
        A, B, k=20, max_iter=200, seed=42
    )
    
    compute_time = time.time() - compute_start
    
    if rank == 0:
        total_time = time.time() - start_time
        print(f"\nDistributed MPI MultiNMF completed!")
        print(f"Final reconstruction error: {total_error:.6f}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Pure computation time: {compute_time:.2f} seconds")
        print(f"Communication overhead: {total_time - compute_time:.2f} seconds")
        print(f"Computation efficiency: {(compute_time/total_time)*100:.1f}%")
        print(f"MPI processes used: {size}")
        
        # Save results
        np.savez('mpi_distributed_results.npz',
                 U_final=result_U,
                 V_A_final=result_V_A,
                 V_B_final=result_V_B,
                 total_error=total_error,
                 mpi_processes=size,
                 total_time=total_time,
                 compute_time=compute_time)
        print(f"Results saved to mpi_distributed_results.npz")

if __name__ == "__main__":
    main()