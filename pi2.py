from mpi4py import MPI

import numpy as np
import time
import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def sample(nsamples):
    pi = 0.0
    hits = 0

    for i in range(nsamples):
        x = random.random()
        y = random.random()
        if ((x * x) + (y * y) < 1):
            hits += 1

    mypi = 1 * np.arange(1, dtype=np.float32)
    mypi[0] = (hits * 1.0) / nsamples
    total = np.empty(mypi.shape, dtype = mypi.dtype)
    comm.Allreduce(mypi, total, op = MPI.SUM)
    pi = (4.0 / size) * total[0]

    return pi

if __name__ == '__main__':
    start = time.process_time()
    pi = sample(1000000 // size)
    end = time.process_time()
    if (comm.Get_rank() == 0):
        print("pi =", pi, "\nпроцессы:", size, "\nвремя:", end - start)