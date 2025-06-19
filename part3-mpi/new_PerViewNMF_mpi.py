import numpy as np
import numpy.matlib
import time

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    def mpi_matmul(A, B):
        """Distributed matrix multiplication with fallback"""
        if size == 1:
            return A @ B
        
        # For small matrices, compute on master and broadcast
        if A.shape[0] < 100:
            if rank == 0:
                result = A @ B
            else:
                result = None
            return comm.bcast(result, root=0)
        
        # Distribute A by rows
        rows_per_proc = A.shape[0] // size
        extra_rows = A.shape[0] % size
        
        if rank < extra_rows:
            start_row = rank * (rows_per_proc + 1)
            local_rows = rows_per_proc + 1
        else:
            start_row = rank * rows_per_proc + extra_rows
            local_rows = rows_per_proc
        
        end_row = start_row + local_rows
        local_A = A[start_row:end_row, :] if local_rows > 0 else np.empty((0, A.shape[1]), dtype=A.dtype)
        
        # Broadcast B and compute local result
        B = comm.bcast(B, root=0)
        local_result = local_A @ B if local_A.size > 0 else np.empty((0, B.shape[1]), dtype=A.dtype)
        
        # Gather and broadcast final result
        all_results = comm.gather(local_result, root=0)
        if rank == 0:
            result = np.vstack([r for r in all_results if r.size > 0])
        else:
            result = None
        return comm.bcast(result, root=0)
    
except ImportError:
    MPI_AVAILABLE = False
    rank = 0
    size = 1
    mpi_matmul = lambda A, B: A @ B
	
def PerViewNMF(X = None,k = None,Vo = None,options = None,U = None,V = None): 
	# This is a module of Multi-View Non-negative Matrix Factorization
# (MultiNMF) for the update for one view as in lines 5-9 in Alg. 1
	
	# Notation:
# X ... (mFea x nSmp) data matrix of one view
#	   mFea  ... number of features
#	   nSmp  ... number of samples
# k ... number of hidden factors
# Vo... consunsus
# options ... Structure holding all settings
# U ... initialization for basis matrix
# V ... initialization for coefficient matrix
	
	#   Originally written by Deng Cai (dengcai AT gmail.com) for GNMF
#   Modified by Jialu Liu (jliu64@illinois.edu)

	differror = options["error"];
	maxIter = options["maxIter"];
	nRepeat = options["nRepeat"];
	minIterOrig = options["minIter"];
	minIter = minIterOrig - 1;
	meanFitRatio = options["meanFitRatio"];

	bSuccess = {}	
	#differror = options.error
	#maxIter = options.maxIter
	#nRepeat = options.nRepeat
	#minIterOrig = options.minIter
	#minIter = minIterOrig - 1
	#meanFitRatio = options.meanFitRatio
	#alpha = options.alpha
	alpha = options["alpha"]
	bSuccess["bSuccess"] = 1

	Norm = 1
	NormV = 0
	mFea,nSmp = X.shape
	selectInit = 1
	norms = None
	#U = None
	#V = None

	if U.shape[0]==0:
		U = np.abs(np.random.rand(mFea,k))
		V = np.abs(np.random.rand(nSmp,k))
		if rank == 0:
			print("Entering PerViewNMF, U is null")
	else:
		nRepeat = 1
	
	U,V = Normalize(U,V)
	if nRepeat == 1:
		selectInit = 0
		minIterOrig = 0
		minIter = 0
		#if len(maxIter)==0:
		if maxIter==None or maxIter==0: #might not be run===========
			objhistory = CalculateObj(X=X,U=U,V=V,L=Vo,alpha=alpha)
			meanFit = objhistory * 10
		else:
			#if isfield(options,'Converge') and options.Converge:
			if "Converge" in options and options["Converge"]:
				objhistory = CalculateObj(X=X,U=U,V=V,L=Vo,alpha=alpha)
	else:
		#if isfield(options,'Converge') and options.Converge:
		if "Converge" in options and options["Converge"]:
			raise Exception('Not implemented!')
	
	tryNo = 0

	while tryNo < nRepeat:
		tmp_T = time.time()
		#tmp_T = cputime
		tryNo = tryNo + 1
		nIter = 0
		maxErr = 1
		nStepTrial = 0
		#disp a
		X = np.array(X, dtype=np.float32)
		U = np.array(U, dtype=np.float32)
		V = np.array(V, dtype=np.float32)

		while (maxErr > differror):

			# ===================== update V ========================
			# ORIGINAL: XU = np.matmul(np.transpose(X), U)
			# ORIGINAL: UU = np.matmul(np.transpose(U), U)
			# ORIGINAL: VUU = np.matmul(V, UU)
			# MODIFIED WITH MPI:
			XU = mpi_matmul(np.transpose(X), U)
			UU = mpi_matmul(np.transpose(U), U)
			VUU = mpi_matmul(V, UU)
			XU = XU + np.multiply(alpha, Vo)
			VUU = VUU + np.multiply(alpha, V)
			V = np.multiply(V, np.divide(XU, np.maximum(VUU,1e-10)))
			# ===================== update U ========================
			# ORIGINAL: XV = np.matmul(X, V)
			# ORIGINAL: VV = np.matmul(np.transpose(V), V)
			# ORIGINAL: UVV = np.matmul(U, VV)
			# MODIFIED WITH MPI:
			XV = mpi_matmul(X, V)
			VV = mpi_matmul(np.transpose(V), V)
			UVV = mpi_matmul(U, VV)
			VV_ = np.matlib.repmat(np.multiply(np.transpose(np.diag(VV)),np.sum(U, axis=0)),mFea,1)
			tmp = np.sum(np.multiply(V,Vo), 0)
			VVo = np.matlib.repmat(tmp,mFea,1)
			XV = XV + np.multiply(alpha, VVo)
			UVV = UVV + np.multiply(alpha, VV_)
			U = np.multiply(U, np.divide(XV, np.maximum(UVV,1e-10)))
			U,V = Normalize(U,V)
			nIter = nIter + 1
			if nIter > minIter:
				if selectInit:
					objhistory = CalculateObj(X=X,U=U,V=V,L=Vo,alpha=alpha)
					maxErr = 0
				else:
					#if len(maxIter)==0:
					if maxIter==None or maxIter==0: #might not be run===========
						newobj = CalculateObj(X=X,U=U,V=V,L=Vo,alpha=alpha)
						objhistory = np.array([objhistory,newobj])
						meanFit = meanFitRatio * meanFit + (1 - meanFitRatio) * newobj
						maxErr = (meanFit - newobj) / meanFit
					else:
						#if isfield(options,'Converge') and options.Converge:
						if "Converge" in options and options["Converge"]:
							newobj = CalculateObj(X=X,U=U,V=V,L=Vo,alpha=alpha)
							objhistory = np.array([objhistory,newobj])
						maxErr = 1
						if nIter >= maxIter:
							maxErr = 0
							#if isfield(options,'Converge') and options.Converge:
							if "Converge" in options and options["Converge"]:
								pass
							else:
								objhistory = 0

		#elapse = cputime - tmp_T
		elapse = time.time() - tmp_T
		if tryNo == 1:
			U_final = U
			V_final = V
			nIter_final = nIter
			elapse_final = elapse
			objhistory_final = objhistory
			bSuccess["nStepTrial"] = nStepTrial
		else:
			if (not isinstance(objhistory, list) and objhistory < objhistory_final) or \
			(isinstance(objhistory,list) and objhistory[-1] < objhistory_final[-1]):
			#if objhistory(end()) < objhistory_final(end()):
				U_final = U
				V_final = V
				nIter_final = nIter
				objhistory_final = objhistory
				#bSuccess.nStepTrial = nStepTrial
				bSuccess["nStepTrial"] = nStepTrial
				if selectInit:
					elapse_final = elapse
				else:
					elapse_final = elapse_final + elapse
		if selectInit:
			if tryNo < nRepeat:
				#re-start
				U = np.abs(np.random.rand(mFea,k))
				V = np.abs(np.random.rand(nSmp,k))
				U,V = Normalize(U,V)
				if rank == 0:
					print("Entering PerViewNMF, selectInit is true, tryNo < nRepeat")
			else:
				tryNo = tryNo - 1
				minIter = 0
				selectInit = 0
				U = U_final
				V = V_final
				objhistory = objhistory_final
				meanFit = objhistory * 10

	
	nIter_final = nIter_final + minIterOrig
	U_final,V_final = Normalize(U_final,V_final)
	return U_final,V_final,nIter_final,elapse_final,bSuccess,objhistory_final
	#==========================================================================
	
	
def CalculateObj(X = None,U = None,V = None,L = None,alpha = None,deltaVU = None,dVordU = None): 
	if deltaVU is None:
		deltaVU = 0
	
	if dVordU is None:
		dVordU = 1
	
	dV = None
	maxM = 62500000
	mFea,nSmp = X.shape
	mn = np.asarray(X).size
	nBlock = int(np.floor(mn * 3 / maxM))
	obj_NMF = None
	dX = None
	if mn < maxM:
		dX = np.matmul(U, np.transpose(V)) - X
		obj_NMF = np.sum(np.square(dX))
		if deltaVU:
			if dVordU:
				dV = np.matmul(np.transpose(dX), U) + np.matmul(L, V)
			else:
				dV = np.matmul(dX, V)
	else:
		obj_NMF = 0
		if deltaVU:
			if dVordU:
				dV = np.zeros((V.shape[0],V.shape[1]))
			else:
				dV = np.zeros((U.shape[0],U.shape[1]))

		iter1 = int(np.ceil(nSmp/nBlock))+1
		print(iter1)
		for i in range(1, iter1):
			if i == iter1-1:
				smpIdx = np.arange((i-1)*nBlock, nSmp, dtype=int)
			else:
				smpIdx = np.arange((i-1)*nBlock, i*nBlock, dtype=int)
			dX = np.matmul(U, np.transpose(V[smpIdx,:])) - X[:,smpIdx]
			obj_NMF = obj_NMF + np.sum(np.square(dX))
			if deltaVU:
				if dVordU:
					dV[smpIdx,:] = np.matmul(np.transpose(dX), U)
				else:
					dV = dU + np.matmul(dX, V[smpIdx,:])
		print("Done")
		if deltaVU:
			if dVordU:
				dV = dV + np.matmul(L, V)
	tmp = V - L
	obj_Lap = np.sum(np.square(tmp))
	dX = np.matmul(U, np.transpose(V)) - X
	obj_NMF = np.sum(np.square(dX))
	obj = obj_NMF + alpha * obj_Lap
	return obj
	
def Normalize(U = None,V = None): 
	U,V = NormalizeUV(U,V,0,1)
	return U, V
	
def NormalizeUV(U = None,V = None,NormV = None,Norm = None): 
	nSmp = V.shape[0]
	mFea = U.shape[0]
	if Norm == 2:
		if NormV:
			norms = np.sqrt(np.sum(np.square(V), axis=0))
			norms = np.maximum(norms,1e-10)
			V = np.divide(V, np.matlib.repmat(norms,nSmp,1))
			U = np.multiply(U, np.matlib.repmat(norms,mFea,1))
			#V = V / np.matlib.repmat(norms,nSmp,1)
			#U = np.multiply(U,np.matlib.repmat(norms,mFea,1))
		else:
			norms = np.sqrt(np.sum(np.square(U), axis=0))
			norms = np.maximum(norms,1e-10)
			U = np.divide(U, np.matlib.repmat(norms,mFea,1))
			V = np.multiply(V,np.matlib.repmat(norms,nSmp,1))
			#norms = np.sqrt(np.sum(U ** 2, 1-1))
			#norms = np.amax(norms,1e-10)
			#U = U / np.matlib.repmat(norms,mFea,1)
			#V = np.multiply(V,np.matlib.repmat(norms,nSmp,1))
	else:
		if NormV:
			norms = np.sum(np.abs(V), axis=0)
			norms = np.maximum(norms,1e-10)
			#V = V / np.matlib.repmat(norms,nSmp,1)
			#U = np.multiply(U,np.matlib.repmat(norms,mFea,1))
			V = np.divide(V, np.matlib.repmat(norms,nSmp,1))
			U = np.multiply(U,np.matlib.repmat(norms,mFea,1))
		else:
			norms = np.sum(np.abs(U), axis=0)
			norms = np.maximum(norms,1e-10)
			U = np.divide(U, np.matlib.repmat(norms,mFea,1))
			#V = np.multiply(V,np.matlib.repmat(norms,nSmp,1))
			#V = bsxfun(times,V,norms)
			V = np.multiply(V, np.matlib.repmat(norms,nSmp,1))
	return U, V
	#return U_final,V_final,nIter_final,elapse_final,bSuccess,objhistory_final
