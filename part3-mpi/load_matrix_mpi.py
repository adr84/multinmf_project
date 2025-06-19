import os

os.environ["MKL_NUM_THREADS"] = "1"  # Set to 1 for MPI to avoid conflicts
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except ImportError:
    MPI_AVAILABLE = False
    rank = 0
    size = 1
    comm = None

import scipy
import numpy as np
import scipy.io
from new_NMF import NMF
from new_PerViewNMF import PerViewNMF
import datetime
import sys

options = {}
options["maxIter"] = 200; #modify 200
options["error"] = 1e-6;
options["nRepeat"] = 30; #modify 30
options["minIter"] = 50; #modify 50
options["meanFitRatio"] = 0.1;
options["rounds"] = 30;
options["alpha"] = [0.01, 0.01]
Rounds = options["rounds"]
num_factor = 20

# Only master process loads data
if rank == 0:
	if MPI_AVAILABLE:
		print(f"MPI MultiNMF starting with {size} processes")
	
	mat = scipy.io.loadmat(sys.argv[1])
	data = {}

	expr_T = np.transpose(mat["expr"])
	morpho = -1.0*mat["morpho"]
	#morpho = mat["morpho"]

	print("Matrix size")
	print("expr_T", expr_T.shape)
	print("morpho", morpho.shape)
	Min = np.abs(np.min(np.min(morpho, axis=0)))
	morpho = morpho + Min
	print(Min)

	data[0] = np.transpose(expr_T)
	data[1] = np.transpose(morpho)

	print(data[0].shape)
	print(data[1].shape)
	#data[0] = np.transpose(mat["expr"])
	#data[1] = np.transpose(mat["morpho"])

	for i in range(2):
		data[i] = data[i] / np.sum(data[i])
else:
	data = None

# Broadcast data to all processes if MPI available
if MPI_AVAILABLE:
	data = comm.bcast(data, root=0)
	if rank != 0:
		print(f"Process {rank}: Received data - A: {data[0].shape}, B: {data[1].shape}")

U_final = {0:np.array([]), 1:np.array([])}
V_final = {0:np.array([]), 1:np.array([])}
V_centroid = {0:np.array([]), 1:np.array([])}
U_ = np.array([])
V_ = np.array([])
U = {0:np.array([]), 1:np.array([])}
V = {0:np.array([]), 1:np.array([])}
bSuccess = {}

if rank == 0:
	print(data[0].shape)
	print(data[1].shape)
#sys.exit(0)

if rank == 0:
	print("init 0...", datetime.datetime.now())
U[0],V[0],nIter,elapse,bSuccess,objhistory = NMF(X=data[0], k=num_factor, options=options, bSuccess=bSuccess, U_=U_, V_=V_)
if rank == 0:
	print("init 1...", datetime.datetime.now())
U[1],V[1],nIter,elapse,bSuccess,objhistory = NMF(X=data[1], k=num_factor, options=options, bSuccess=bSuccess, U_=U_, V_=V[0])
if rank == 0:
	print("init 2...", datetime.datetime.now())
U[0],V[0],nIter,elapse,bSuccess,objhistory = NMF(X=data[0], k=num_factor, options=options, bSuccess=bSuccess, U_=U_, V_=V[1])
if rank == 0:
	print("init 3...", datetime.datetime.now())
U[1],V[1],nIter,elapse,bSuccess,objhistory = NMF(X=data[1], k=num_factor, options=options, bSuccess=bSuccess, U_=U_, V_=V[0])
if rank == 0:
	print("init 4...", datetime.datetime.now())
U[0],V[0],nIter,elapse,bSuccess,objhistory = NMF(X=data[0], k=num_factor, options=options, bSuccess=bSuccess, U_=U_, V_=V[1])
if rank == 0:
	print("init 5...", datetime.datetime.now())
U[1],V[1],nIter,elapse,bSuccess,objhistory = NMF(X=data[1], k=num_factor, options=options, bSuccess=bSuccess, U_=U_, V_=V[0])


optionsForPerViewNMF = options.copy()
oldL = 100
#tic
j = 0
log = []
centroidV = None
while j < Rounds:
	j = j + 1
	if j == 1:
		centroidV = V[0]
	else:
		centroidV = np.multiply(options["alpha"][0], V[0])
		for i in range(1, 2):
			centroidV = centroidV + np.multiply(options["alpha"][i],V[i])
		centroidV = np.divide(centroidV, np.sum(options["alpha"]))
	logL = 0

	for i in range(2):
		tmp1 = data[i] - np.matmul(U[i], np.transpose(V[i]))
		tmp2 = V[i] - centroidV
		logL = logL + np.sum(np.square(tmp1)) + options["alpha"][i] * np.sum(np.square(tmp2))

	log.append(logL)
	if rank == 0:
		print(logL)
	if (oldL < logL):
		U = oldU
		V = oldV
		logL = oldL
		j = j - 1
		if rank == 0:
			print('restrart this iteration')
	oldU = U
	oldV = V
	oldL = logL
	for i in range(2):
		#print(options["alpha"][i])
		#print(optionsForPerViewNMF["alpha"])
		optionsForPerViewNMF["alpha"] = options["alpha"][i]
		U[i],V[i],_,_,_,_ = PerViewNMF(X=data[i],k=num_factor,Vo=centroidV,options=optionsForPerViewNMF,U=U[i],V=V[i])

# Only master process saves results
if rank == 0:
	np.savez(sys.argv[2], U_final=U, V_final=V, V_centroid=centroidV)
	if MPI_AVAILABLE:
		print(f"MPI MultiNMF completed using {size} processes")
	print(f"Results saved to {sys.argv[2]}")
