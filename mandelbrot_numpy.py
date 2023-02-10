from matplotlib import pyplot as plt
import time
import numpy

pRE = 1000
pIM = 1000
threshold = 2


def color_it(t):
    return int(255 * (t / 200)), int(255 * (t / 60)), int(255 * (t / 20))


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

    div_time = numpy.zeros(c.shape, dtype=int)

    # Iterate over the complex plane
    for i in range(1000):
        # Apply the Mandelbrot formula
        z[mandelbrot_mask] = z[mandelbrot_mask] * z[mandelbrot_mask] + c[mandelbrot_mask]

        # Check each element of the array for divergence
        diverged = mandelbrot_mask & (numpy.abs(z) > threshold)
        # Update the divergence time
        div_time[diverged] = i

        # Check if the absolute value of z is greater than the threshold
        mandelbrot_mask[numpy.abs(z) > threshold] = False

    return mandelbrot_mask, div_time


def main():
    start_time = time.time()

    # Generates linear spaces with 300 and 200 elements respectively around the plane of the Mandelbrot set
    x_space = numpy.linspace(-2.5, 1.5, 300)
    y_space = numpy.linspace(-1.5, 1.5, 200)

    # Generate a 2D array for each dimension
    complete_space = x_space[:, numpy.newaxis] + y_space[numpy.newaxis, :] * 1j
    # Apply the Mandelbrot formula
    complete_space, div_time = mandelbrot(complete_space)

    print("Complete space:", complete_space)
    print("Div time:", div_time)

    print("Computation time:", time.time() - start_time)
    plt.imshow(div_time, cmap='magma')
    # plt.imshow(div_time)
    plt.show()


if __name__ == '__main__':
    main()
    # Computation time: 0.42s

