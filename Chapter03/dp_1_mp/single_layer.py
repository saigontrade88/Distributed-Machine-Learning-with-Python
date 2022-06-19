import os
import sys
from datetime import datetime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from utils import rank_print, setup_distrib
import torchvision
from torchvision import datasets, transforms
from torch import optim

DESTINATION_RANK=0
OPERATION=dist.ReduceOp.SUM
DSET_FOLDER = "./cifar10_data"

def net_setup():
	os.environ['MASTER_ADDR'] = '127.0.0.1'
	os.environ['MASTER_PORT'] = '5000'

def main_process(rank, world_size=3):
	print(f"Process for rank: {rank} has been spawned")
	# Setup the distributed processing
	device = setup_distrib(rank, world_size)
	
	# Load the dataset in all processes download only in the first one
	if rank == 0:
		dset = torchvision.datasets.CIFAR10(DSET_FOLDER, download=True)
	
	# Make sure download has finished
	dist.barrier()
	
	# Load the dataset
	dset = torchvision.datasets.CIFAR10(DSET_FOLDER)
	input_size = 3 * 32 * 32 # [channel size, height, width]
	per_gpu_batch_size = 128
	num_classes = 10
	
	#Initialze the model
	if dist.get_rank() == 0:
		weights = torch.rand((input_size, num_classes), device=device)
	else:
		weights = torch.zeros((input_size, num_classes), device=device)

	# Distribute weights to all GPUs. Thus, the same model weights among all the processes
	handle = dist.broadcast(tensor=weights, src=0, async_op=True)
	rank_print(f"Weights received.")

	# Flattened images
	cur_input = torch.zeros((per_gpu_batch_size, input_size), device=device)
	# One-Hot encoded target
	cur_target = torch.zeros((per_gpu_batch_size, num_classes), device=device)

	for i in range(per_gpu_batch_size):
		rank_print(f"Loading image {i+ per_gpu_batch_size*rank} into GPU...")
		image, target = dset[i+ per_gpu_batch_size*rank]
		cur_input[i] = transforms.ToTensor()(image).flatten()
		cur_target[i, target] = 1.0 #example

	# Compute the linear part of the layer
	output = torch.matmul(cur_input, weights)
	rank_print(f"\nComputed output: {output}, Size: {output.size()}.")

	# Define the activation function of the output layer
	logsoftm = torch.nn.LogSoftmax(dim=1)
	
	# Apply log softmax activation function to output layer for predictin
	output = logsoftm(output)
	rank_print(f"\nLog-Softmaxed output: {output}, Size: {output.size()}.")
	loss = output.sum(dim=1).mean()
	rank_print(f"Loss: {loss}, Size: {loss.size()}")
	
	# Here the GPUs need to be synched again
	#dist.reduce(tensor=loss, dst=DESTINATION_RANK, op=dist.ReduceOp.SUM)
	dist.all_reduce(tensor=loss, op=OPERATION)
	rank_print(f"Final Total Loss: {loss}")
	rank_print(f"Final Average Loss: {loss/world_size}")
	
if __name__ == "__main__":
	#nprocs=number gpus
	start_time = datetime.now() 
	print("Begin at {}".format(start_time.strftime('%Y-%m-%d %H:%M:%S')))
	net_setup()
	#Use Pytorch mul-processing to spawn processes
	mp.spawn(main_process, nprocs=3, args=())
	now = datetime.now()
	print("Complete at {}".format(now.strftime('%Y-%m-%d %H:%M:%S')))
	print("--- {} seconds ---".format(now - start_time))
	sys.stdout.flush()