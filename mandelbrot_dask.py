from matplotlib import pyplot as plt
import time
import numpy
import multiprocessing
import dask
from dask.distributed import Client
import dask.array as da
import numpy

import mandel_naive_numpy

threshold = 2
iterations = 100

performance_metrics = []


def mandelbrot(c):
    """
    Generate a Mandelbrot set using vectorized numpy operations.

    :param c:
    :return mandelbrot:
    """
    # Generate a 2D array of ones, which is then converted to a boolean data type array
    mandelbrot_mask = numpy.ones_like(c, dtype=bool)
    # Generate a 2D array of zeros, which is then converted to a complex data type array
    z = numpy.zeros_like(c, dtype=complex)
    # z is iteratively updated with the Mandelbrot formula: z = z^2 + c

    divergence_time = numpy.zeros(c.shape, dtype=int)

    # Iterate over the complex plane
    for i in range(iterations):
        # Apply the Mandelbrot formula
        z[mandelbrot_mask] = z[mandelbrot_mask] * z[mandelbrot_mask] + c[mandelbrot_mask]

        # Check each element of the array for divergence
        diverged = mandelbrot_mask & (numpy.abs(z) > threshold)
        # Update the divergence time
        divergence_time[diverged] = i

        # Check if the absolute value of z is greater than the threshold
        mandelbrot_mask[numpy.abs(z) > threshold] = False

    return divergence_time


def dask_datatype(pRE, pIM, chunk_size, show_figure=True):
    start_time = time.time()

    # Generates linear spaces with pRE and pIM elements respectively around the plane of the Mandelbrot set
    x_space = numpy.linspace(-2.3, 0.8, pRE).reshape((1, pRE))
    y_space = numpy.linspace(-1.2, 1.2, pIM).reshape((pIM, 1))

    # Generate a 2D array for each dimension of the complex plane
    complete_space = x_space + y_space * 1j
    complete_space = da.from_array(complete_space)

    """
    client = Client(processes=True)
    print("Client created:", client)
    solution_return = client.map(mandelbrot, complete_space)
    print("Solution returned:", solution_return)
    solution_return = client.gather(solution_return)

    print("Solution:", solution_return)
    """

    solution_return = complete_space.map_blocks(mandelbrot, dtype=numpy.int32).compute()
    end_time = time.time()
    print("Chunk size:", chunk_size, "Computation time:", round(end_time-start_time,3),"s")

    if show_figure:
        plt.imshow(solution_return, cmap='magma')
        plt.show()


if __name__ == '__main__':
    print("Comparing performance of numpy and dask:")
    print("Numpy:")
    mandel_naive_numpy.main(3000, 3000, show_figure=False)
    print("Dask:")
    dask_datatype(3000, 3000, (3000, 3000), show_figure=False)

    print("\n\nComparing DASK with different chunk sizes:")
    chunk_sizes = [(1000, 1000), (500, 500), (200, 200), (100, 100), (50, 50), (25, 25), (10, 10), (5, 5)]
    for s_chunk in chunk_sizes:
        dask_datatype(1000, 1000, s_chunk, show_figure=False)
