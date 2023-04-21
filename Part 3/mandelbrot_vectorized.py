"""
    Author: Lukas Bisgaard Kristensen
    Date: 26. April 2023
    Course: Numerical Scientific Computing, AAU
    Description: This program computes the Mandelbrot set using OpenCL.
"""

import matplotlib.pyplot as plt
import time
import numpy
import doctest


def mandelbrot(c):
    """
    Computes the Mandelbrot set using vectorized numpy operations.

    :param c: Input complex array
    :return mandelbrot: Divergence time

    Usage examples:
    >>> mandelbrot(numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) * 1j)
    array([0., 0., 1., 0., 0., 0., 0., 0., 0., 0.], dtype=float16)
    >>> mandelbrot(numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) + numpy.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) * 1j)
    array([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.], dtype=float16)
    """
    # Generate a 2D array of ones, which is then converted to a boolean data type array
    mandelbrot_mask = numpy.ones_like(c, dtype=bool)

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

    # doctest.testmod()
    return divergence_time


def generate_space(pRE, pIM, data_type):
    """
    Generates the space of the Mandelbrot set.

    :param pRE: Real part
    :param pIM: Imaginary part
    :param data_type: Data type of the array
    :return: space

    Usage examples:
    >>> generate_space(1000, 1000, numpy.float16)
    array([[-2.3007812-1.2001953j, -2.296875 -1.2001953j,
            -2.2929688-1.2001953j, ...,  0.7939453-1.2001953j,
             0.796875 -1.2001953j,  0.7998047-1.2001953j],
           [-2.3007812-1.1972656j, -2.296875 -1.1972656j,
            -2.2929688-1.1972656j, ...,  0.7939453-1.1972656j,
             0.796875 -1.1972656j,  0.7998047-1.1972656j],
           [-2.3007812-1.1953125j, -2.296875 -1.1953125j,
            -2.2929688-1.1953125j, ...,  0.7939453-1.1953125j,
             0.796875 -1.1953125j,  0.7998047-1.1953125j],
           ...,
           [-2.3007812+1.1953125j, -2.296875 +1.1953125j,
            -2.2929688+1.1953125j, ...,  0.7939453+1.1953125j,
             0.796875 +1.1953125j,  0.7998047+1.1953125j],
           [-2.3007812+1.1972656j, -2.296875 +1.1972656j,
            -2.2929688+1.1972656j, ...,  0.7939453+1.1972656j,
             0.796875 +1.1972656j,  0.7998047+1.1972656j],
           [-2.3007812+1.2001953j, -2.296875 +1.2001953j,
            -2.2929688+1.2001953j, ...,  0.7939453+1.2001953j,
             0.796875 +1.2001953j,  0.7998047+1.2001953j]], dtype=complex64)
    """

    # Generates linear spaces with pRE and pIM elements respectively around the plane of the Mandelbrot set
    x_space = numpy.linspace(-2.3, 0.8, pRE, dtype=data_type).reshape((1, pRE))
    y_space = numpy.linspace(-1.2, 1.2, pIM, dtype=data_type).reshape((pIM, 1))

    # Generate a 2D array for each dimension of the complex plane
    generated_space = x_space + y_space * 1j
    return generated_space


def computation_time(start_time, end_time):
    """
    Computes the time taken to compute the Mandelbrot set.

    :param start_time: Start time of computation
    :param end_time: End time of computation
    :return: Difference between the end time and the start time

    Usage examples:
    >>> computation_time(0, 0.792)
    0.792
    """
    return round(end_time - start_time, 3)


def main(pRE, pIM, show_figure=True):
    """
    :param pRE: Real part
    :param pIM: Imaginary part
    :param show_figure: Condition if matplotlib should show the figure
    :return: Computation time
    """

    start_time = time.time()

    complete_space = generate_space(pRE, pIM, numpy.float16)

    # Apply the Mandelbrot formula
    computed_mandelbrot = mandelbrot(complete_space)

    end_time = time.time()

    time_computed = computation_time(start_time, end_time)
    print("Computation time:", time_computed,"s")

    if show_figure:
        plt.imshow(computed_mandelbrot, cmap='magma')
        plt.show()
    return round(end_time - start_time, 3)


if __name__ == '__main__':
    doctest.testmod(report=True, verbose=True)
    main(1000, 1000)

