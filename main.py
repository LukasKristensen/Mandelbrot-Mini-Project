from matplotlib import pyplot as plt

pRE = 10000
pIM = 10000
threshold = 2


def mandelbrot(x, y):
    c = complex(x, y)
    z = 0

    for t in range(1, 10000):
        if abs(z) > threshold:
            return int(255*(t/200)), int(255*(t/60)), int(255*(t/30))
        z = z*z + c
    return 255, 255, 255


if __name__ == '__main__':
    solution = [[(0, 0, 0) for x in range(pRE)] for y in range(pIM)]

    for x in range(pRE):
        for y in range(pIM):
            solution[y][x] = mandelbrot((x-(pRE*0.75))/(pRE*0.35), (y-(pRE*0.5))/(pRE*0.35))

    plt.imshow(solution)
    plt.show()

