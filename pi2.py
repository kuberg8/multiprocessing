from mpi4py import MPI
from termcolor import cprint

import numpy as np
import time
import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def getPi(nsamples):
    pi = 0.0
    hits = 0

    for i in range(nsamples):
        x = random.random()
        y = random.random()

        if ((x * x) + (y * y) < 1):
            hits += 1

    mypi = 1 * np.arange(1, dtype = np.float32) # получения картежа с одним значением дробных чисел
    mypi[0] = (hits * 1.0) / nsamples # вычисление среза
    total = np.empty(mypi.shape, dtype = mypi.dtype) # создание картежа по форме переменной mypi с 0 значением

    # https://mpitutorial.com/tutorials/mpi-reduce-and-allreduce/
    comm.Allreduce(mypi, total, op = MPI.SUM) # вычисление(сложение total) из всех процессов
    pi = (4.0 / size) * total[0]

    return pi

if __name__ == '__main__':
    start = time.process_time()
    pi = getPi(1000000 // size)
    end = time.process_time()

    if (rank == 0):
        cprint(f'pi = {pi}\nпроцессы: {size}\nвремя: {end - start}', "blue")