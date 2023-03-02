import numba
import numpy
from matplotlib import pyplot as plt
import time
import numba

pRE = 1000
pIM = 1000
threshold = 2


def mandelbrot(data_type):
    start_time = time.time()

    # Generates linear spaces with pRE and pIM elements respectively around the plane of the Mandelbrot set
    x_space = numpy.linspace(-2.3, 0.8, pRE, dtype=data_type).reshape((1, pRE))
    y_space = numpy.linspace(-1.2, 1.2, pIM, dtype=data_type).reshape((pIM, 1))

    # Generate a 2D array for each dimension of the complex plane
    complete_space = x_space + y_space * 1j

    # Generate a 2D array of zeroes, which is then converted to a boolean data type array
    mandelbrot_mask = numpy.ones_like(complete_space, dtype=bool)
    z = numpy.zeros_like(complete_space, dtype=complex)
    divergence_time = numpy.zeros(complete_space.shape, dtype=data_type)

    # Iterate over the complex plane
    for i in range(100):
        # Apply the Mandelbrot formula
        z[mandelbrot_mask] = z[mandelbrot_mask] * z[mandelbrot_mask] + complete_space[mandelbrot_mask]

        # Check each element of the array for divergence
        diverged = mandelbrot_mask & (numpy.abs(z) > threshold)
        # Update the divergence time
        divergence_time[diverged] = i

        # Check if the absolute value of z is greater than the threshold
        mandelbrot_mask[numpy.abs(z) > threshold] = False
    print("Data-type:",data_type,"Computation time:", time.time() - start_time)
    return divergence_time


def main(show_figure=False):
    mandelbrot(numpy.float64)
    mandelbrot(numpy.float32)
    mandelbrot(numpy.float16)

    if show_figure:
        plt.imshow(mandelbrot(numpy.float64), cmap='magma')
        plt.show()


if __name__ == '__main__':
    main()
    # Computation time: 1.39s

