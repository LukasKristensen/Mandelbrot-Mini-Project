from matplotlib import pyplot as plt
import time
import numpy
import multiprocessing

pRE = 1000
pIM = 1000
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


def main(chunk_size, cores, show_figure=True):
    start_time = time.time()

    # Generates linear spaces with pRE and pIM elements respectively around the plane of the Mandelbrot set
    x_space = numpy.linspace(-2.3, 0.8, pRE).reshape((1, pRE))
    y_space = numpy.linspace(-1.2, 1.2, pIM).reshape((pIM, 1))

    # Generate a 2D array for each dimension of the complex plane
    complete_space = x_space + y_space * 1j

    pool_workers = multiprocessing.Pool(processes=cores)
    solution_return = pool_workers.map(mandelbrot, complete_space, chunksize=chunk_size)
    reshaped_solution = numpy.array(solution_return).reshape(pIM, pRE)

    print("Computation time:", round(time.time() - start_time, 5), "Cores:", cores, "Chunk Size:", chunk_size)
    performance_metrics.append((cores, chunk_size, time.time() - start_time))

    if show_figure:
        plt.imshow(reshaped_solution, cmap='magma')
        plt.show()


if __name__ == '__main__':
    """
    Run this file to generate a 3D plot of the computation time in relation to the number of cores and chunk size.
    """

    for core in range(1, multiprocessing.cpu_count() + 1):
        for chunk_size in range(1, 200, 10):
            main(chunk_size=chunk_size, cores=core, show_figure=False)

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot_trisurf([x[0] for x in performance_metrics], [x[1] for x in performance_metrics], [x[2] for x in performance_metrics], edgecolor='none', cmap='viridis', antialiased=False)

    ax.set_xlabel('Cores')
    ax.set_ylabel('Chunk Size')
    ax.set_zlabel('Computation Time')

    plt.show()

    # Without multiprocessing
    # Computation time: 2.12s

    # 1 Processor
    # Computation time: 2.7245500087738037

    # 2 Processors
    # Computation time: 1.861546277999878s

    # 4 Processors
    # Computation time: 1.6297404766082764s

    # 8 Processors
    # Computation time: 1.5058352947235107s

    # -----------------------------
    # 8 Processors with Chunk Size 1
    # 1.6212193965911865s

    # 8 Processors with Chunk Size 5
    # Computation time: 1.5571467876434326s

    # 8 Processors with Chunk Size 10
    # Computation time: 1.6028096675872803s

    # 8 Processors with Chunk Size 100
    # Computation time: 1.5139856338500977s

    # 8 Processors with Chunk Size 200
    # Computation time: 1.7952868938446045s

    # 8 Processors with Chunk Size 500
    # Computation time: 2.2340340614318848s

