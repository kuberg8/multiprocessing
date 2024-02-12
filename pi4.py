from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

def compute_pi(n):
    count = n // size
    start = rank * count
    end = (rank + 1) * count

    if rank == size - 1:
        end = n

    subtotal = 0

    for i in range(start, end):
        x = (i + 0.5) / n
        subtotal += 4 / (1 + x ** 2)

    pi = MPI.COMM_WORLD.reduce(subtotal, op=MPI.SUM, root=0)

    if rank == 0:
        pi /= n

    return pi

if __name__ == '__main__':
    if rank == 0:
        print('Введите количество итераций: ')
        n = int(input())
    else:
        n = None

    n = MPI.COMM_WORLD.bcast(n, root=0)
    pi = compute_pi(n)

    if rank == 0:
        print("Вычисленное значение числа Пи:", pi)