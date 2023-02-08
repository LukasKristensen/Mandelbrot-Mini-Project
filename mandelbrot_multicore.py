import multiprocessing
from matplotlib import pyplot as plt
import time

pRE = 10
pIM = 10
threshold = 2


def color_it(t):
    print("Returning color", t)
    return int(255 * (t / 200)), int(255 * (t / 60)), int(255 * (t / 20))


def mandelbrot(x, y):
    c = complex(x, y)
    z = 0

    for i in range(100):
        if abs(z) > threshold:
            return color_it(i)
        z = z*z + c
    return 255


start_time, solution = 0, 0


def test(y):
    global solution
    solution = [[(0, 0, 0) for m in range(pRE)] for n in range(pIM)]

    for x in range(pRE):
        print("Old val:",x,y,solution[x][y])
        solution[x][y] = mandelbrot((x - (pRE * 0.75)) / (pRE * 0.35), (y - (pRE * 0.5)) / (pRE * 0.35))
        print("New val:",x,y,solution[x][y])

    return solution


def main():
    global start_time

    mp_pool = multiprocessing.Pool(processes=4)
    received_solution = mp_pool.map_async(test, range(pIM), chunksize=1)
    mp_pool.close()
    mp_pool.join()

    collected_array = []
    print("Solution:", received_solution)
    for value in received_solution.get():
        collected_array.append(value[0])

    print("Collected array:", collected_array)
    for i in collected_array:
        print(i)

    plt.imshow(collected_array)
    plt.show()


if __name__ == '__main__':
    start_time = time.time()

    main()

    # Computation time: 4.46s
    print("Computation time:", time.time() - start_time)



