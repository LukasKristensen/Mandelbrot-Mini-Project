import numpy
from matplotlib import pyplot as plt
import time

pRE = 1000
pIM = 1000
threshold = 2


def color_it(t):
    return int(255 * (t / 200)), int(255 * (t / 60)), int(255 * (t / 20))


def mandelbrot(x, y):
    c = complex(x, y)
    z = 0

    for i in range(100):
        if abs(z) > threshold:
            return i
        z = z*z + c
    return 255


def main():
    start_time = time.time()
    solution = numpy.vecotorize(numpy.zeros(shape=(pIM, pRE, 3), dtype=numpy.uint16))

    for x in range(pRE):
        for y in range(pIM):
            solution[y][x] = color_it(mandelbrot((x-(pRE*0.75))/(pRE*0.35), (y-(pRE*0.5))/(pRE*0.35)))

    print("Computation time:", time.time() - start_time)
    plt.imshow(solution)
    plt.show()


if __name__ == '__main__':
    main()
    # Computation time: 4.78s

