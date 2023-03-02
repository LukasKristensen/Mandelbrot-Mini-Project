import numba
import numpy
from matplotlib import pyplot as plt
import time

pRE = 1000
pIM = 1000
threshold = 2


def mandelbrot(z, c):
    for i in range(100):
        if abs(z) > threshold:
            return i
        z = z*z + c
    return 0


def main(show_figure=True):
    # Integer
    start_time = time.time()
    solution = numpy.array(pRE * pIM * [0], dtype=numpy.int64).reshape(pRE, pIM)
    for x in range(pRE):
        for y in range(pIM):
            solution[y][x] = mandelbrot(numpy.int64(0), complex(numpy.int64((x-(pRE*0.75))/(pRE*0.35))+1j*numpy.int64((y-(pRE*0.5))/(pRE*0.35))))
    print("[Integer] Computation time:", time.time() - start_time)

    # Float64
    start_time = time.time()
    solution = numpy.array(pRE * pIM * [0], dtype=numpy.float64).reshape(pRE, pIM)
    for x in range(pRE):
        for y in range(pIM):
            solution[y][x] = mandelbrot(numpy.float64(0), complex(numpy.float64((x-(pRE*0.75))/(pRE*0.35))+1j*numpy.float64((y-(pRE*0.5))/(pRE*0.35))))
    print("[Float64] Computation time:", time.time() - start_time)

    # Float32
    start_time = time.time()
    solution = numpy.array(pRE * pIM * [0], dtype=numpy.float32).reshape(pRE, pIM)
    for x in range(pRE):
        for y in range(pIM):
            solution[y][x] = mandelbrot(numpy.float32(0), complex(numpy.float32((x-(pRE*0.75))/(pRE*0.35))+1j*numpy.float32((y-(pRE*0.5))/(pRE*0.35))))
    print("[Float32] Computation time:", time.time() - start_time)

    # Float16
    start_time = time.time()
    solution = numpy.array(pRE * pIM * [0], dtype=numpy.float16).reshape(pRE, pIM)
    for x in range(pRE):
        for y in range(pIM):
            solution[y][x] = mandelbrot(numpy.float16(0), complex(numpy.float16((x-(pRE*0.75))/(pRE*0.35))+1j*numpy.float16((y-(pRE*0.5))/(pRE*0.35))))
    print("[Float16] Computation time:", time.time() - start_time)

    if show_figure:
        plt.imshow(solution, cmap='magma')
        plt.show()


if __name__ == '__main__':
    main()
    # Computation time: 1.39s

