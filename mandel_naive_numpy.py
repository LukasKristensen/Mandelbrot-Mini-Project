import numpy
from matplotlib import pyplot as plt
import time

threshold = 2


def mandelbrot(x, y):
    """
    Compute the Mandelbrot set using a naive approach
    :param x:
    :param y:
    :return:
    """
    c = complex(x, y)
    z = 0

    for i in range(100):
        if abs(z) > threshold:
            return i
        z = z*z + c
    return 0


def main(pRE, pIM, show_figure=True):
    start_time = time.time()
    solution = numpy.empty((pRE, pIM, 1), dtype=numpy.uint8)

    for x in range(pRE):
        for y in range(pIM):
            solution[y][x] = mandelbrot((x-(pRE*0.75))/(pRE*0.35), (y-(pRE*0.5))/(pRE*0.35))

    print("Computation time:", time.time() - start_time)

    if show_figure:
        plt.imshow(solution, cmap='magma')
        plt.show()


if __name__ == '__main__':
    main(1000, 1000)
    # Computation time: 3.91s

