from matplotlib import pyplot as plt
import time
import numpy
import cv2
import os
import datetime
import multiprocessing

pRE = 1000
pIM = 1000
threshold = 2
iterations = 500

frame_rate = 60
step_size = frame_rate*61


def mandelbrot(c):
    """
    Generate a Mandelbrot set using vectorized numpy operations.

    :param c:
    :return mandelbrot:
    """
    # Generate a 2D array of ones, which is then converted to a boolean data type array
    mandelbrot_mask = numpy.ones_like(c, dtype=bool)
    # Generate a 2D array of zeros, which is then converted to a complex data type array
    z = numpy.zeros_like(c, dtype=numpy.complex128)
    # z is iteratively updated with the Mandelbrot formula: z = z^2 + c

    divergence_time = numpy.zeros(c.shape, dtype=numpy.float64)

    # Iterate over the complex plane
    for i in range(iterations):
        # Apply the Mandelbrot formula
        z[mandelbrot_mask] = z[mandelbrot_mask] * z[mandelbrot_mask] + c[mandelbrot_mask]

        # Check each element of the array for divergence
        diverged = mandelbrot_mask & (numpy.abs(z) > threshold)
        # Update the divergence time
        divergence_time[diverged] = i

        # Check if the absolute value of z is greater than the threshold
        mandelbrot_mask[numpy.abs(z) > threshold] = False

    return divergence_time


def main(x_0, y_0, x_1, y_1):
    print("X:",x_0, x_1, "Y:", y_0, y_1, "")
    start_time = time.time()

    # Generates linear spaces with pRE and pIM elements respectively around the plane of the Mandelbrot set
    x_space = numpy.linspace(x_0, x_1, pRE, dtype=numpy.float64).reshape((1, pRE))
    y_space = numpy.linspace(y_0, y_1, pIM, dtype=numpy.float64).reshape((pIM, 1))

    # Generate a 2D array for each dimension of the complex plane
    complete_space = x_space + y_space * 1j

    # Apply the Mandelbrot formula
    pool_workers = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    solution_return = pool_workers.map(mandelbrot, complete_space, chunksize=100)
    reshaped_solution = numpy.array(solution_return).reshape(pIM, pRE)

    print("Frame computation time:", time.time() - start_time)

    return reshaped_solution


if __name__ == '__main__':
    output_video_destination = f'mandelbrot_animation_{str(datetime.datetime.now()).replace(":", "-").replace(".", "")}.avi'
    print(f'Video save destination: {os.getcwd()}/{output_video_destination}')
    start_render_time = time.time()

    start_point = [-2.25, 0.75, -1.25, 1.25]
    end_point = [-0.7336438924199521-(4.5E-14)/2, -0.7336438924199521+(4.5E-14)/2, 0.2455211406714035-(4.5E-14)/2, 0.2455211406714035+(4.5E-14)/2]
    
    video_writer = cv2.VideoWriter(output_video_destination, cv2.VideoWriter_fourcc(*'DIVX'), frame_rate, (pRE, pIM))

    for i in range(step_size):
        t = 1-(1-i / step_size)**10
        x0, x1, y0, y1 = start_point
        a0, a1, b0, b1 = end_point

        current_x = [x0 + t * (a0 - x0), x1 + t * (a1 - x1)]
        current_y = [y0 + t * (b0 - y0), y1 + t * (b1 - y1)]

        # Apply the Mandelbrot formula and generate the image
        complete_mandelbrot = main(current_x[0], current_y[0], current_x[1], current_y[1])

        plt.figure(figsize=(pRE, pIM), dpi=1)
        plt.imshow(complete_mandelbrot, cmap='magma', interpolation='nearest', aspect='auto')
        plt.axis('off')
        plt.bbox_inches = 'tight'
        plt.pad_inches = 0
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(f'tmp_hold.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        loaded_image = cv2.imread('tmp_hold.png')
        cv2.putText(loaded_image, f'X: {round(current_x[0], 4)} : {round(current_x[1], 4)}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(loaded_image, f'Y: {round(current_y[0], 4)} : {round(current_y[0], 4)}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        video_writer.write(loaded_image)

        # Progress bar and time estimation
        progress_count = round((i / step_size) * 100, 2)
        print(f'\n\nProgress: {progress_count}%')
        if i != 0:
            print("Remaining time:", round((time.time() - start_render_time) * (1 - (i / step_size)) / (i / step_size), 2), "seconds")
        plt.pause(0.001)  # Pause for 1ms to allow the plot to update

    # Hold the last frame for 3 seconds when the video ends
    for i in range(frame_rate*3):
        video_writer.write(cv2.imread('tmp_hold.png'))
    video_writer.release()
    os.remove('tmp_hold.png')  # Remove the temporary image file
    print("Video saved to:", os.getcwd() + "/" + output_video_destination)
    print("Complete! Total render time:", time.time() - start_render_time)
