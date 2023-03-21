import numpy
from matplotlib import pyplot as plt
import time


pRE = 10000
pIM = 10000
iterations = 100
threshold = 2


def mandelbrot(float_data_type, complex_data_type):
    start_time = time.time()

    # Generates linear spaces with pRE and pIM elements respectively around the plane of the Mandelbrot set
    x_space = numpy.linspace(-2.3, 0.8, pRE, dtype=float_data_type).reshape((1, pRE))
    y_space = numpy.linspace(-1.2, 1.2, pIM, dtype=float_data_type).reshape((pIM, 1))

    # Generate a 2D array for each dimension of the complex plane
    complete_space = x_space + y_space * 1j

    # Generate a 2D array of zeroes, which is then converted to a boolean data type array
    mandelbrot_mask = numpy.ones_like(complete_space, dtype=bool)
    z = numpy.zeros_like(complete_space, dtype=complex_data_type)
    divergence_time = numpy.zeros(complete_space.shape, dtype=float_data_type)

    # Iterate over the complex plane
    for i in range(iterations):
        # Apply the Mandelbrot formula
        z[mandelbrot_mask] = z[mandelbrot_mask] * z[mandelbrot_mask] + complete_space[mandelbrot_mask]

        # Check each element of the array for divergence
        diverged = mandelbrot_mask & (numpy.abs(z) > threshold)
        # Update the divergence time
        divergence_time[diverged] = i

        # Check if the absolute value of z is greater than the threshold
        mandelbrot_mask[numpy.abs(z) > threshold] = False
    end_time = time.time()
    print("Float data-type:", float_data_type, "Complex data-type:", complex_data_type,
          "Computation time:", round(end_time - start_time, 5), "s")
    return round(end_time - start_time, 5)


def main(show_figure=False):
    mandelbrot(numpy.float64, numpy.complex64)
    mandelbrot(numpy.float64, numpy.complex128)
    mandelbrot(numpy.float32, numpy.complex64)
    mandelbrot(numpy.float32, numpy.complex128)
    mandelbrot(numpy.float16, numpy.complex64)
    mandelbrot(numpy.float16, numpy.complex128)


if __name__ == '__main__':
    main()

