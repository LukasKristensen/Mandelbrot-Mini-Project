from matplotlib import pyplot as plt
import time
import numpy
import multiprocessing

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


def main(chunk_size, cores, pRE, pIM, show_figure=True, x0=-2.3, x1=0.8, y0=-1.2, y1=1.2):
    start_time = time.time()

    # Generates linear spaces with pRE and pIM elements respectively around the plane of the Mandelbrot set
    x_space = numpy.linspace(x0, x1, pRE).reshape((1, pRE))
    y_space = numpy.linspace(y0, y1, pIM).reshape((pIM, 1))

    # Generate a 2D array for each dimension of the complex plane
    complete_space = x_space + y_space * 1j

    pool_workers = multiprocessing.Pool(processes=cores)
    solution_return = pool_workers.map(mandelbrot, complete_space, chunksize=chunk_size)
    reshaped_solution = numpy.array(solution_return).reshape(pIM, pRE)

    end_time = time.time()
    print("Computation time:", round(end_time - start_time, 3), "s")
    performance_metrics.append((cores, chunk_size, time.time() - start_time))

    if show_figure:
        plt.imshow(reshaped_solution, cmap='magma')
        plt.show()

    return reshaped_solution


if __name__ == '__main__':
    """
    Run this file to generate a 3D plot of the computation time in relation to the number of cores and chunk size.
    """

    for core in range(1, multiprocessing.cpu_count() + 1):
        for chunk_size in range(1, 200, 10):
            main(chunk_size=chunk_size, cores=core, pRE=1000, pIM=1000, show_figure=False)

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot_trisurf([x[0] for x in performance_metrics], [x[1] for x in performance_metrics], [x[2] for x in performance_metrics], edgecolor='none', cmap='viridis', antialiased=False)

    ax.set_xlabel('Cores')
    ax.set_ylabel('Chunk Size')
    ax.set_zlabel('Computation Time')

    plt.show()
