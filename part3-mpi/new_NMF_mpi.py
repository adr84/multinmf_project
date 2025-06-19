import numpy as np
import numpy.matlib
import time
import os

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

def NMF(X=None, k=None, options=None, bSuccess=None, U_=None, V_=None):
	differror = options["error"];
	maxIter = options["maxIter"];
	nRepeat = options["nRepeat"];
	minIterOrig = options["minIter"];
	minIter = minIterOrig - 1;
	meanFitRatio = options["meanFitRatio"];

	bSuccess["bSuccess"] = 1
	mFea, nSmp = X.shape
	Norm = 1
	NormV = 0
	selectInit = 1
	norms = None
	U = None
	V = None
	meanFit = 0

	#k is number of hidden factors
	if U_.shape[0]==0:
		U = np.abs(np.random.rand(mFea, k)) #random numbers of 0...1 for a matrix of shape mFea and k
		norms = np.sqrt(np.sum(np.square(U), axis=0))
		norms = np.maximum(norms, 1e-10)
		U = np.divide(U, np.matlib.repmat(norms, mFea, 1))
		if rank == 0:
			print("Entering NMF, U_ is null")

		if V_.shape[0]==0:
			V = np.abs(np.random.rand(nSmp, k))
			V = np.divide(V, np.sum(V))
			if rank == 0:
				print("Entering NMF, U_ is null, V_ is null")
		else:
			V = V_
	else:
		U = U_
		if V_.shape[0]==0:
			V = np.abs(np.random.rand(nSmp, k))
			V = np.divide(V, np.sum(V))
			if rank == 0:
				print("Entering NMF, V_ is null")
		else:
			V = V_

	U,V = NormalizeUV(U=U,V=V,NormV=NormV,Norm=Norm)
	if nRepeat == 1:
		selectInit = 0
		minIterOrig = 0
		minIter = 0
		#if len(maxIter)==0:
		if maxIter==None or maxIter==0: #might not be run===========
			objhistory = CalculateObj(X=X,U=U,V=V)
			meanFit = objhistory * 10
		else:
			#check that Converge exists ============
			#if isfield(options,'Converge') and options.Converge:
			if "Converge" in options and options["Converge"]:
				objhistory = CalculateObj(X=X,U=U,V=V)
	else:
		#check that Converge exists ===============
		#if isfield(options,'Converge') and options.Converge:
		if "Converge" in options and options["Converge"]:
			raise Exception('Not implemented!')

	if rank == 0:
		print("nRepeat is", nRepeat)

	tryNo = 0
	while tryNo < nRepeat:
		#might not exist
		#tmp_T = cputime
		tmp_T = time.time()
		tryNo = tryNo + 1
		nIter = 0
		maxErr = 1
		nStepTrial = 0
		if rank == 0:
			print("NMF tryNo", tryNo)
		#X = np.asfortranarray(X)
		#U = np.asfortranarray(U)
		#V = np.asfortranarray(V)
		X = np.array(X, dtype=np.float32)
		U = np.array(U, dtype=np.float32)
		V = np.array(V, dtype=np.float32)

		while (maxErr > differror):

			#print("NMF maxErr", maxErr, "nIter", nIter)
			# ===================== update V ========================
			# ORIGINAL: V = np.multiply(V, (X.T @ U) / np.maximum(V @ (U.T @ U), 1e-10))
			# MODIFIED WITH MPI:
			XTU = mpi_matmul(X.T, U)
			UTU = mpi_matmul(U.T, U)
			VUTU = mpi_matmul(V, UTU)
			V = np.multiply(V, XTU / np.maximum(VUTU, 1e-10))
			
			# ===================== update U ========================
			# ORIGINAL: U = np.multiply(U, (X @ V) / np.maximum(U @ (V.T @ V), 1e-10))
			# MODIFIED WITH MPI:
			XV = mpi_matmul(X, V)
			VTV = mpi_matmul(V.T, V)
			UVTV = mpi_matmul(U, VTV)
			U = np.multiply(U, XV / np.maximum(UVTV, 1e-10))
			
			nIter = nIter + 1

			#print("nIter", nIter, "minIter", minIter)
			if nIter > minIter:
				if selectInit:
					if rank == 0:
						print("selectInit", selectInit, nIter, meanFit)
					objhistory = CalculateObj(X=X,U=U,V=V)
					#newobj
					meanFit = meanFitRatio * meanFit + (1 - meanFitRatio) * objhistory
					if rank == 0:
						print("done")
					maxErr = 0
				else:
					#if len(maxIter)==0:
					if maxIter==None or maxIter==0: #might not be run===========
						newobj = CalculateObj(X=X,U=U,V=V)
						objhistory = np.array([objhistory,newobj])
						meanFit = meanFitRatio * meanFit + (1 - meanFitRatio) * newobj
						maxErr = (meanFit - newobj) / meanFit
					else:
						#check following line is true=======================
						#if isfield(options,'Converge') and options.Converge:
						if "Converge" in options and options["Converge"]:
							newobj = CalculateObj(X=X,U=U,V=V)
							objhistory = np.array([objhistory,newobj])
						maxErr = 1
						if nIter >= maxIter:
							maxErr = 0
							#check following line is true========================
							#if isfield(options,'Converge') and options.Converge:
							if "Converge" in options and options["Converge"]:
								pass
							else:
								objhistory = 0

		#check following line is true
		#elapse = cputime - tmp_T
		elapse = time.time() - tmp_T
		if tryNo == 1:
			U_final = U
			V_final = V
			nIter_final = nIter
			elapse_final = elapse
			objhistory_final = objhistory
			#bSuccess.nStepTrial = nStepTrial #check this line
			bSuccess["nStepTrial"] = nStepTrial
		else:
			#print(objhistory)
			#print(objhistory_final)
			
			if (not isinstance(objhistory, list) and objhistory < objhistory_final) or \
			(isinstance(objhistory,list) and objhistory[-1] < objhistory_final[-1]):
				U_final = U
				V_final = V
				nIter_final = nIter
				objhistory_final = objhistory
				#bSuccess.nStepTrial = nStepTrial #check this true
				bSuccess["nStepTrial"] = nStepTrial
				if selectInit:
					elapse_final = elapse
				else:
					elapse_final = elapse_final + elapse
			#else:
			#	print("You should not pass here")
		#print("selectInit", selectInit)
		if selectInit:
			if tryNo < nRepeat:
				#re-start
				if U_.shape[0]==0:
					U = np.abs(np.random.rand(mFea,k))
					norms = np.sqrt(np.sum(np.square(U), axis=0))
					norms = np.maximum(norms,1e-10)
					U = np.divide(U, np.matlib.repmat(norms,mFea,1))
					if rank == 0:
						print("Entering NMF, selectInit is true, U_ is null")
					if V_.shape[0]==0:
						V = np.abs(np.random.rand(nSmp,k))
						V = np.divide(V, np.sum(V))
						if rank == 0:
							print("Entering NMF, selectInit is true, U_ is null, V_ is null")
					else:
						V = V_
				else:
					U = U_
					if V_.shape[0]==0:
						V = np.abs(np.random.rand(nSmp,k))
						V = np.divide(V, np.sum(V))
						if rank == 0:
							print("Entering NMF, selectInit is true, V_ is null")
					else:
						V = V_
				U,V = NormalizeUV(U=U,V=V,NormV=NormV,Norm=Norm)
			else:
				tryNo = tryNo - 1
				minIter = 0
				selectInit = 0
				U = U_final
				V = V_final
				objhistory = objhistory_final
				meanFit = objhistory * 10

	
	nIter_final = nIter_final + minIterOrig
	U_final,V_final = Normalize(U=U_final,V=V_final)

	return U_final,V_final,nIter_final,elapse_final,bSuccess,objhistory_final
	#==========================================================================
	
	
def CalculateObj(X = None,U = None,V = None,deltaVU = None,dVordU = None): 
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
		#print("Here0")
		dX = np.matmul(U, np.transpose(V)) - X
		obj_NMF = np.sum(np.square(dX))
		if deltaVU:
			if dVordU:
				dV = np.matmul(np.transpose(dX), U)
			else:
				dV = np.matmul(dX, V)
		#print(dX)
		#print("Done0")
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
		#for i in np.arange(1, np.ceil(nSmp / nBlock)+1).reshape(-1):
			if i == iter1-1:
				#smpIdx = np.arange((i - 1) * nBlock + 1,nSmp+1)
				smpIdx = np.arange((i-1)*nBlock, nSmp, dtype=int)
			else:
				#smpIdx = np.arange((i - 1) * nBlock + 1,i * nBlock+1)
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
				dV = dV
	
	obj = obj_NMF
	#return obj, dV
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
		else:
			norms = np.sqrt(np.sum(np.square(U), axis=0))
			norms = np.maximum(norms,1e-10)
			U = np.divide(U, np.matlib.repmat(norms,mFea,1))
			V = np.multiply(V,np.matlib.repmat(norms,nSmp,1))
	else:
		if NormV:
			norms = np.sum(np.abs(V), axis=0)
			norms = np.maximum(norms,1e-10)
			V = np.divide(V, np.matlib.repmat(norms,nSmp,1))
			U = np.multiply(U,np.matlib.repmat(norms,mFea,1))
		else:
			norms = np.sum(np.abs(U), axis=0)
			#norms = max(norms,1e-10);
			U = np.divide(U, np.matlib.repmat(norms,mFea,1))
			V = np.multiply(V,np.matlib.repmat(norms,nSmp,1))
	return U, V
