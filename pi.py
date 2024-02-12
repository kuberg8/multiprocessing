from mpi4py import MPI
from termcolor import cprint

comm = MPI.COMM_WORLD
rank = comm.Get_rank() # текущий процесс
size = comm.Get_size() # кол-во процессов

slice_size = 1000000 # размер каждого среза работы
total_slices = 50 # общее количество срезов, которые необходимо выполнить

if rank == 0:
    pi = 0
    slice = 0
    process = 1

    # Отправление первой партии процессов для запуска вычислений в каждом процессе
    while process < size and slice < total_slices:
        comm.send(slice, dest=process, tag=1)
        cprint(f'Запуск вычисления и отправка среза - {slice} в процесс - {process} из процесса - {rank}', "yellow")
        slice += 1
        process += 1

    # Ожидание пока выполнятся расчеты
    received_processes = 0
    while received_processes < total_slices:
        # MPI.ANY_SOURCE - получение данных с любого источника(процесса)
        pi += comm.recv(source = MPI.ANY_SOURCE, tag = 1)
        process = comm.recv(source = MPI.ANY_SOURCE, tag = 2)
        cprint(f'Получены данные pi - {pi} от процесса - {process}; срез - {received_processes}', "green")
        received_processes += 1

        if slice < total_slices:
            comm.send(slice, dest=process, tag = 1)
            cprint(f'Отправка среза - {slice} в процесс - {process}', "blue")
            slice += 1

    # Отправление значения(-1) для остановки каждого процесса вычисления
    for process in range(1, size):
        comm.send(-1, dest=process, tag = 1)

    print("Pi =", 4.0 * pi)

# Это процессы, где ранг > 0. Они выполняют вычисления.
else:
    while True:
        start = comm.recv(source = 0, tag = 1)
        if start == -1:
            cprint(f'Остановка процесса - {rank}', 'red')
            break

        i = 0
        slice_value = 0

        while i < slice_size:
            if i % 2 == 0:
                slice_value += 1.0 / (2 * (start * slice_size + i) + 1)
            else:
                slice_value -= 1.0 / (2 * (start * slice_size + i) + 1)
            i += 1

        comm.send(slice_value, dest = 0, tag = 1) # отправка результата вычисления в первый(нулевой) процесс
        comm.send(rank, dest = 0, tag = 2) # отправка значения в каком процессе производились вычисления