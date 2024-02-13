from mpi4py import MPI
from termcolor import cprint

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def compute_pi(n):
    # Выбор промежутока итераций для текущего процесса
    count = n // size
    start = rank * count
    end = (rank + 1) * count

    if rank == size - 1:
        end = n
    ########

    subtotal = 0

    for i in range(start, end):
        x = (i + 0.5) / n
        subtotal += 4 / (1 + x ** 2)

    # https://mpitutorial.com/tutorials/mpi-reduce-and-allreduce/
    pi = comm.reduce(subtotal, op=MPI.SUM, root=0)

    if rank == 0:
        pi /= n

    return pi

if __name__ == '__main__':
    if rank == 0:
        cprint('Введите количество итераций:', 'blue')
        n = int(input())
    else:
        n = None

    n = comm.bcast(n, root=0)
    pi = compute_pi(n)

    if rank == 0:
        cprint(f'Вычисленное значение числа Пи: {pi}', 'green')