from matplotlib import pyplot as plt
import time
import numpy


def mandelbrot(c):
    """
    Generate a Mandelbrot set using vectorized numpy operations.

    :param c:
    :return mandelbrot:
    """
    # Generate a 2D array of ones, which is then converted to a boolean data type array
    mandelbrot_mask = numpy.ones_like(c, dtype=numpy.bool)

    # Generate a 2D array of zeros, which is then converted to a complex data type array
    z = numpy.zeros_like(c, dtype=numpy.complex64)

    divergence_time = numpy.zeros(c.shape, dtype=numpy.float16)

    # Iterate over the complex plane
    for i in range(100):
        # Apply the Mandelbrot formula
        z[mandelbrot_mask] = z[mandelbrot_mask] * z[mandelbrot_mask] + c[mandelbrot_mask]

        # Check each element of the array for divergence
        diverged = mandelbrot_mask & (numpy.abs(z) > 2)
        # Update the divergence time
        divergence_time[diverged] = i

        # Check if the absolute value of z is greater than the threshold
        mandelbrot_mask[numpy.abs(z) > 2] = False

    return divergence_time


def main(pRE, pIM, show_figure=True):
    start_time = time.time()

    # Generates linear spaces with pRE and pIM elements respectively around the plane of the Mandelbrot set
    x_space = numpy.linspace(-2.3, 0.8, pRE, dtype=numpy.float16).reshape((1, pRE))
    y_space = numpy.linspace(-1.2, 1.2, pIM, dtype=numpy.float16).reshape((pIM, 1))

    # Generate a 2D array for each dimension of the complex plane
    complete_space = x_space + y_space * 1j

    # Apply the Mandelbrot formula
    computed_mandelbrot = mandelbrot(complete_space)

    end_time = time.time()
    print("Size:", pRE, pIM, "Computation time:", round(end_time-start_time, 3), "s")

    if show_figure:
        plt.imshow(computed_mandelbrot, cmap='magma')
        plt.show()


if __name__ == '__main__':
    main(1000, 1000)

