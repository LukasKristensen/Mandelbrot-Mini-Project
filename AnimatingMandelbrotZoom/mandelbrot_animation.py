from matplotlib import pyplot as plt
import time
import numpy
import numba

pRE = 600
pIM = 600
threshold = 2


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

    divergence_time = numpy.zeros(c.shape, dtype=numpy.float32)

    # Iterate over the complex plane
    for i in range(100):
        # Apply the Mandelbrot formula
        z[mandelbrot_mask] = z[mandelbrot_mask] * z[mandelbrot_mask] + c[mandelbrot_mask]

        # Check each element of the array for divergence
        diverged = mandelbrot_mask & (numpy.abs(z) > threshold)
        # Update the divergence time
        divergence_time[diverged] = i

        # Check if the absolute value of z is greater than the threshold
        mandelbrot_mask[numpy.abs(z) > threshold] = False

    return divergence_time


def main(x_0, y_0, x_1, y_1):
    print("X:",x_0, x_1, "Y:", y_0, y_1, "")
    start_time = time.time()

    # Generates linear spaces with pRE and pIM elements respectively around the plane of the Mandelbrot set
    x_space = numpy.linspace(x_0, x_1, pRE, dtype=numpy.float32).reshape((1, pRE))
    y_space = numpy.linspace(y_0, y_1, pIM, dtype=numpy.float32).reshape((pIM, 1))

    # Generate a 2D array for each dimension of the complex plane
    complete_space = x_space + y_space * 1j

    # Apply the Mandelbrot formula
    computed_mandelbrot = mandelbrot(complete_space)

    print("Computation time:", time.time() - start_time)

    return computed_mandelbrot


if __name__ == '__main__':
    # Coordinates to interpolate between
    # First four coordinates are the start and end points of the interpolation
    # The fifth coordinate is the number of steps to take between the start and end points
    interest_points = [[-2.25, 0.75, -1.25, 1.25, 30],
                       [-0.352917, -0.127973, -0.722195, -0.534797, 30],
                       [-0.206791, -0.195601, -0.68154, -0.672274, 30],
                       [-0.19925116, -0.199134805, -0.679549605, -0.67945249, 30]]
    for z in range(1, len(interest_points)):
        start_x, start_y = (interest_points[z-1][0], interest_points[z-1][1]), (interest_points[z-1][2], interest_points[z-1][3])
        target_x = (interest_points[z][0], interest_points[z][1])
        target_y = (interest_points[z][2], interest_points[z][3])
        step_size = interest_points[z][4]

        dif_x, dif_y = (target_x[0]-start_x[0], target_x[1]-start_x[1]), (target_y[0]-start_y[0], target_y[1]-start_y[1])
        step_x, step_y = (dif_x[0]/step_size, dif_x[1]/step_size), (dif_y[0]/step_size, dif_y[1]/step_size)

        print("Differences:", dif_x, dif_y)
        print("Steps:", step_x, step_y)
        print("\n\n")

        for i in range(step_size+1):
            print("Applying step calculations:",step_x[0]*i, step_x[1]*i, step_y[0]*i, step_y[1]*i)
            current_x = (start_x[0]+step_x[0]*i, start_x[1]+step_x[1]*i)
            current_y = (start_y[0]+step_y[0]*i, start_y[1]+step_y[1]*i)

            computed_mandelbrot = main(current_x[0], current_y[0], current_x[1], current_y[1])
            plt.imshow(computed_mandelbrot, cmap='magma')
            plt.pause(0.01)
    plt.show()

