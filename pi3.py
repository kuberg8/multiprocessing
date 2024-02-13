from mpi4py import MPI
import random

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def compute_pi(num_points):
    points_inside_circle = 0
    points_per_process = num_points // size
    local_points_inside_circle = 0

    for _ in range(points_per_process):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)

        distance = x**2 + y**2

        if distance <= 1:
            local_points_inside_circle += 1

    points_inside_circle = comm.reduce(local_points_inside_circle, op=MPI.SUM, root=0)
    total_points = num_points

    if rank == 0:
        pi = 4 * (points_inside_circle / total_points)
        return pi


if __name__ == "__main__":
    num_points = 10000000
    pi_approximation = compute_pi(num_points)

    if (rank == 0):
        print("Pi =", pi_approximation)