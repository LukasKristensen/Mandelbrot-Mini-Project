import numba
from matplotlib import pyplot as plt
import time

pRE = 1000
pIM = 1000
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


def main():
    start_time = time.time()
    solution = [[(0, 0, 0) for x in range(pRE)] for y in range(pIM)]

    for x in range(pRE):
        for y in range(pIM):
            solution[y][x] = mandelbrot((x-(pRE*0.75))/(pRE*0.35), (y-(pRE*0.5))/(pRE*0.35))

    print("Computation time:", time.time() - start_time)
    plt.imshow(solution, cmap='magma')
    plt.show()


if __name__ == '__main__':
    main()
    # Computation time: 1.45s

