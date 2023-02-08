import multiprocessing
from matplotlib import pyplot as plt
import time

pRE = 1000
pIM = 1000
threshold = 2


def color_it(t):
    return int(255 * (t / 200)), int(255 * (t / 60)), int(255 * (t / 20))


solution = [[(0, 0, 0) for x in range(pRE)] for y in range(pIM)]


def mandelbrot(x, y):
    c = complex(x, y)
    z = 0

    for i in range(100):
        if abs(z) > threshold:
            return i
        z = z*z + c
    return 255


start_time, solution = 0, 0


def main(x):
    global start_time
    global solution

    for y in range(pIM):
        try:
            solution[x][y] = mandelbrot((x - (pRE * 0.75)) / (pRE * 0.35), (y - (pRE * 0.5)) / (pRE * 0.35))
        except TypeError:
            print(x, y)


if __name__ == '__main__':
    start_time = time.time()

    print("Count:", multiprocessing.cpu_count())
    mp_pool = multiprocessing.Pool(processes=2)
    mp_pool.map(main, range(pRE), chunksize=1)

    # Computation time: 4.46s
    print("Computation time:", time.time() - start_time)
    plt.imshow(solution)
    plt.show()

