import datetime

import numba
from matplotlib import pyplot as plt
import time

pRE = 10000
pIM = 10000
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


def main(show_figure=True, save_image=False):
    start_time = time.time()
    solution = [[(0, 0, 0) for x in range(pRE)] for y in range(pIM)]

    for x in range(pRE):
        print("Progress: {:.2f}%".format((x/pRE)*100))
        for y in range(pIM):
            solution[y][x] = mandelbrot((x-(pRE*0.75))/(pRE*0.35), (y-(pRE*0.5))/(pRE*0.35))

    print("Computation time:", time.time() - start_time)

    if show_figure and not save_image:
        plt.imshow(solution, cmap='magma')
        plt.show()
    if save_image:
        plt.figure(figsize=(15000,15000), dpi=1)
        plt.imshow(solution, cmap='magma')
        print("Plot size:", plt.gcf().get_size_inches()*plt.gcf().dpi)
        plt.axis('off')
        plt.savefig(f'mandelbrot_numba_{str(datetime.datetime.now()).replace(":","-")}.png', dpi=2, bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    main(save_image=True)
    # Computation time: 1.39s

