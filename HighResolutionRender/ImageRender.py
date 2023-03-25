import datetime

import numpy
from matplotlib import pyplot as plt
import time

map_size = 2000
iterations = 500


def mandelbrot(c):
    # Generate a 2D array of ones, which is then converted to a boolean data type array
    mandelbrot_mask = numpy.ones_like(c, dtype=bool)

    # Generate a 2D array of zeros, which is then converted to a complex data type array
    z = numpy.zeros_like(c, dtype=numpy.complex128)

    divergence_time = numpy.zeros(c.shape, dtype=numpy.float64)

    # Iterate over the complex plane
    for i in range(iterations):
        # Apply the Mandelbrot formula
        z[mandelbrot_mask] = z[mandelbrot_mask] * z[mandelbrot_mask] + c[mandelbrot_mask]

        # Check each element of the array for divergence
        diverged = mandelbrot_mask & (numpy.abs(z) > 2)
        # Update the divergence time
        divergence_time[diverged] = i

        # Check if the absolute value of z is greater than the threshold
        mandelbrot_mask[numpy.abs(z) > 2] = False

    return divergence_time


def main():
    start_time = time.time()

    # Generates linear spaces with pRE and pIM elements respectively around the plane of the Mandelbrot set
    x_space = numpy.linspace(-0.19925116, -0.199134805, map_size, dtype=numpy.float64).reshape((1, map_size))
    y_space = numpy.linspace(-0.679549605, -0.67945249, map_size, dtype=numpy.float64).reshape((map_size, 1))

    # Generate a 2D array for each dimension of the complex plane
    complete_space = x_space + y_space * 1j

    # Apply the Mandelbrot formula
    computed_mandelbrot = mandelbrot(complete_space)

    print("Computation time:", time.time() - start_time)

    print("Solution size:", len(computed_mandelbrot), len(computed_mandelbrot[0]))
    plt.figure(figsize=(map_size, map_size), dpi=1)
    plt.imshow(computed_mandelbrot, cmap='magma', interpolation='nearest', aspect='auto')

    print("Plot size:", plt.gcf().get_size_inches()*plt.gcf().dpi)
    plt.axis('off')
    plt.bbox_inches = 'tight'
    plt.pad_inches = 0
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(f'mandelbrot_numba_{str(datetime.datetime.now()).replace(":","-")}.png')


if __name__ == '__main__':
    main()

