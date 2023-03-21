import numba
from matplotlib import pyplot as plt
import time

threshold = 2


@numba.vectorize
def mandelbrot(x, y):
    c = complex(x, y)
    z = 0

    for i in range(100):
        if abs(z) > threshold:
            return i
        z = z*z + c
    return 0


def main(pRE, pIM, show_figure=True):
    start_time = time.time()
    solution = [[(0, 0, 0) for x in range(pRE)] for y in range(pIM)]

    for x in range(pRE):
        for y in range(pIM):
            solution[y][x] = mandelbrot((x-(pRE*0.75))/(pRE*0.35), (y-(pRE*0.5))/(pRE*0.35))

    end_time = time.time()
    print("Computation time:", round(end_time - start_time, 3), "s")

    if show_figure:
        plt.imshow(solution, cmap='magma')
        plt.show()


if __name__ == '__main__':
    main(1000, 1000)

