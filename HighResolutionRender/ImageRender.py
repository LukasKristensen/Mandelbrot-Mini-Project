import datetime

import numba
from matplotlib import pyplot as plt
import time

map_size = 10000
threshold = 100


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
    solution = [[(0, 0, 0) for x in range(map_size)] for y in range(map_size)]

    for x in range(map_size):
        print("Progress: {:.2f}%".format((x/map_size)*100))
        for y in range(map_size):
            solution[y][x] = mandelbrot((x-(map_size*0.75))/(map_size*0.35), (y-(map_size*0.5))/(map_size*0.35))

    print("Computation time:", time.time() - start_time)

    print("Solution size:", len(solution), len(solution[0]))
    plt.figure(figsize=(map_size, map_size), dpi=1)
    plt.imshow(solution, cmap='magma', interpolation='nearest', aspect='auto')

    print("Plot size:", plt.gcf().get_size_inches()*plt.gcf().dpi)
    plt.axis('off')
    plt.bbox_inches = 'tight'
    plt.pad_inches = 0
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(f'mandelbrot_numba_{str(datetime.datetime.now()).replace(":","-")}.png')


if __name__ == '__main__':
    main()

