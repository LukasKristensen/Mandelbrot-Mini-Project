from matplotlib import pyplot as plt
import time
from dask.distributed import Client
import dask.array as da
import mandelbrot_vectorized
import numpy as np

threshold = 2
iterations = 100


def mandelbrot(c):
    """
    Generate a Mandelbrot set using vectorized numpy operations.

    :param c:
    :return mandelbrot:
    """
    # Generate a 2D array of ones, which is then converted to a boolean data type array
    mandelbrot_mask = da.ones_like(c, dtype=bool)

    # Generate a 2D array of zeros, which is then converted to a complex data type array
    z = da.zeros_like(c, dtype=np.complex128)

    # z is iteratively updated with the Mandelbrot formula: z = z^2 + c
    divergence_time = da.zeros(c.shape, dtype=np.float64)

    # Iterate over the complex plane
    for i in range(iterations):
        # Apply the Mandelbrot formula
        z = z * z + c

        # Check each element of the array for divergence
        diverged = mandelbrot_mask & (np.abs(z) > threshold)

        # Update the divergence time
        divergence_time[diverged] = i

        # Stops early if the absolute value of z is greater than the threshold (point diverged)
        mandelbrot_mask[np.abs(z) > threshold] = False

    return divergence_time


def dask_local_distribution(pRE, pIM, chunk_size, show_figure=True):
    start_time = time.time()

    # Generates linear spaces with pRE and pIM elements respectively around the plane of the Mandelbrot set
    x_space = da.linspace(-2.3, 0.8, pRE, dtype=np.float64).reshape((1, pRE))
    y_space = da.linspace(-1.2, 1.2, pIM, dtype=np.float64).reshape((pIM, 1))
    # Generate a 2D array for each dimension of the complex plane
    complete_space = da.rechunk(x_space + y_space * 1j, chunks=chunk_size)

    solution_return = complete_space.map_blocks(mandelbrot).compute()
    end_time = time.time()

    print("Computation time:", round(end_time - start_time, 3), "s")
    if show_figure:
        plt.imshow(solution_return, cmap='magma')
        plt.show()


def dask_distributed_execution(client, pRE, pIM, chunk_size, show_figure=False):
    start_time = time.time()

    # Generates linear spaces with pRE and pIM elements respectively around the plane of the Mandelbrot set
    x_space = da.linspace(-2.3, 0.8, pRE, dtype=np.float16).reshape((1, pRE))
    y_space = da.linspace(-1.2, 1.2, pIM, dtype=np.float16).reshape((pIM, 1))
    # Generate a 2D array for each dimension of the complex plane
    complete_space = da.rechunk(x_space + y_space * 1j, chunks=chunk_size)

    solution_return = client.compute(complete_space.map_blocks(mandelbrot)).result()
    end_time = time.time()

    print("Chunk size:", chunk_size, "Computation time:", round(end_time-start_time, 3), "s")
    if show_figure:
        plt.imshow(solution_return, cmap='magma')
        plt.show()


def main():
    plot_size = 1000
    fig_show = False
    chunk_sizes = [(500, 500), (200, 200), (100, 100), (50, 50)]

    # print("\nComparing performance of numpy and dask:")
    # print("Numpy:")
    # mandelbrot_vectorized.main(8000, 8000, show_figure=fig_show)
    print("DASK local execution:")
    dask_local_distribution(8000, 8000, (8000, 8000), show_figure=fig_show)

    print("\nComparing local DASK with different chunk sizes:")
    for s_chunk in chunk_sizes:
        dask_local_distribution(plot_size, plot_size, s_chunk, show_figure=fig_show)

    client_distribute = Client()
    print("Creating a client for distributed DASK execution:", client_distribute)
    print("Dashboard for statistics:",client_distribute.dashboard_link)

    print("\nComparing distributed DASK with different chunk sizes:")
    for s_chunk in chunk_sizes:
        print("\nChunk size:", s_chunk)
        dask_distributed_execution(client_distribute, plot_size, plot_size, s_chunk, show_figure=fig_show)

    print("Results:", client_distribute.dashboard_link)
    client_distribute.close()


if __name__ == "__main__":
    main()
