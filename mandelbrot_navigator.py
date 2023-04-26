"""
    Author: Lukas Bisgaard Kristensen
    Date: 26. April 2023
    Course: Numerical Scientific Computing, AAU
    Description: This program computes the Mandelbrot set using OpenCL.
"""

import pyopencl
import time
import cv2 as cv
from mandelbrot_approaches import mandelbrot_opencl


xmin, xmax, ymin, ymax = -2.3, 0.8, -1.2, 1.2
heatmaps = [cv.COLORMAP_HOT, cv.COLORMAP_BONE, cv.COLORMAP_PINK, cv.COLORMAP_INFERNO, cv.COLORMAP_PLASMA, cv.COLORMAP_TWILIGHT, cv.COLORMAP_OCEAN, cv.COLORMAP_MAGMA, cv.COLORMAP_PARULA, cv.COLORMAP_TURBO]
selected_heatmap = cv.COLORMAP_HOT


def update_heatmap(index):
    global selected_heatmap
    try:
        selected_heatmap = heatmaps[index]
    except IndexError:
        print('Index out of bounds')


def draw_window():
    start_time = time.time()
    mandelbrot_figure = mandelbrot_opencl.mandelbrot_opencl(device=device, context=context, queue=queue, width=1000, height=1000, local_size=25, show_figure=False, x_min=xmin, x_max=xmax, y_min=ymin, y_max=ymax)
    # Add opencv text
    font = cv.FONT_HERSHEY_SIMPLEX

    # Controls
    cv.putText(mandelbrot_figure, 'Controls:', (10, 50), font, 0.5, 255, 1, cv.LINE_AA)
    cv.putText(mandelbrot_figure, 'w: up', (10, 100), font, 0.5, 255, 1, cv.LINE_AA)
    cv.putText(mandelbrot_figure, 's: down', (10, 150), font, 0.5, 255, 1, cv.LINE_AA)
    cv.putText(mandelbrot_figure, 'a: left', (10, 200), font, 0.5, 255, 1, cv.LINE_AA)
    cv.putText(mandelbrot_figure, 'd: right', (10, 250), font, 0.5, 255, 1, cv.LINE_AA)
    cv.putText(mandelbrot_figure, 'q: zoom in', (10, 300), font, 0.5, 255, 1, cv.LINE_AA)
    cv.putText(mandelbrot_figure, 'e: zoom out', (10, 350), font, 0.5, 255, 1, cv.LINE_AA)
    cv.putText(mandelbrot_figure, 'r: reset', (10, 400), font, 0.5, 255, 1, cv.LINE_AA)

    # Position
    cv.putText(mandelbrot_figure, f'x {round(xmin, 5)} : {round(xmax, 5)}', (10, 500), font, 0.5, 255, 1, cv.LINE_AA)
    cv.putText(mandelbrot_figure, f'y {round(ymin, 5)} : {round(ymax, 5)}', (10, 550), font, 0.5, 255, 1, cv.LINE_AA)

    # Computation time
    cv.putText(mandelbrot_figure, f'Computation time: {round(time.time() - start_time, 3)} s', (10, 600), font, 0.5, 255, 1, cv.LINE_AA)

    # Switch colormap
    cv.putText(mandelbrot_figure, 'Switch colormap:', (10, 700), font, 0.5, 255, 1, cv.LINE_AA)
    cv.putText(mandelbrot_figure, '0 - 9', (10, 730), font, 0.5, 255, 1, cv.LINE_AA)

    mandelbrot_figure = cv.convertScaleAbs(mandelbrot_figure)
    mandelbrot_figure = cv.applyColorMap(mandelbrot_figure, selected_heatmap)
    cv.imshow('Mandelbrot Set Navigator', mandelbrot_figure)


def main():
    """
    Script for navigating the Mandelbrot set.

    :return:
    """
    global xmin, xmax, ymin, ymax
    draw_window()

    while True:
        y_scale = (ymax - ymin) / 10
        x_scale = (xmax - xmin) / 10

        k = cv.waitKey(1) & 0xFF
        if k == ord('w'):
            ymin -= y_scale
            ymax -= y_scale
        elif k == ord('s'):
            ymin += y_scale
            ymax += y_scale
        elif k == ord('a'):
            xmin -= x_scale
            xmax -= x_scale
        elif k == ord('d'):
            xmin += x_scale
            xmax += x_scale
        elif k == ord('e'):
            ymin -= y_scale
            ymax += y_scale
            xmin -= x_scale
            xmax += x_scale
        elif k == ord('q'):
            ymin += y_scale
            ymax -= y_scale
            xmin += x_scale
            xmax -= x_scale
        elif k == ord('r'):
            xmin, xmax, ymin, ymax = -2.3, 0.8, -1.2, 1.2
        elif k in range(48, 58):
            update_heatmap(k-48)
        else:
            continue
        draw_window()


if __name__ == '__main__':
    context, queue, device, name = mandelbrot_opencl.create_opencl_context(pyopencl.get_platforms()[0])
    main()
