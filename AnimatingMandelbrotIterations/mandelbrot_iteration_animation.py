from matplotlib import pyplot as plt
import time
import numpy
import cv2
import os

pRE = 700
pIM = 700
threshold = 2
iterations = 100

frames_between_points = 30
frame_rate = round(frames_between_points/6)

output_video_destination = f'mandelbrot_animation_{iterations}.avi'
print(f'Video save destination: {os.getcwd()}/{output_video_destination}')
video_writer = cv2.VideoWriter(output_video_destination, cv2.VideoWriter_fourcc(*'DIVX'), frame_rate, (pRE, pIM))


def mandelbrot(c, iterations_get):
    """
    Generate a Mandelbrot set using vectorized numpy operations.
    :param iterations_get:
    :param c:
    :return mandelbrot:
    """
    # Generate a 2D array of ones, which is then converted to a boolean data type array
    mandelbrot_mask = numpy.ones_like(c, dtype=bool)
    # Generate a 2D array of zeros, which is then converted to a complex data type array
    z = numpy.zeros_like(c, dtype=complex)
    # z is iteratively updated with the Mandelbrot formula: z = z^2 + c

    divergence_time = numpy.zeros(c.shape, dtype=numpy.float32)

    # Iterate over the complex plane
    for i in range(iterations_get):
        # Apply the Mandelbrot formula
        z[mandelbrot_mask] = z[mandelbrot_mask] * z[mandelbrot_mask] + c[mandelbrot_mask]

        # Check each element of the array for divergence
        diverged = mandelbrot_mask & (numpy.abs(z) > threshold)
        # Update the divergence time
        divergence_time[diverged] = i

        # Check if the absolute value of z is greater than the threshold
        mandelbrot_mask[numpy.abs(z) > threshold] = False
        save_frame(divergence_time, i)

    return divergence_time


def main(iterations_get):
    start_time = time.time()

    # Generates linear spaces with pRE and pIM elements respectively around the plane of the Mandelbrot set
    x_space = numpy.linspace(-2.3, 0.8, pRE, dtype=numpy.float32).reshape((1, pRE))
    y_space = numpy.linspace(-1.2, 1.2, pIM, dtype=numpy.float32).reshape((pIM, 1))

    # Generate a 2D array for each dimension of the complex plane
    complete_space = x_space + y_space * 1j

    # Apply the Mandelbrot formula
    computed_mandelbrot = mandelbrot(complete_space, iterations_get)

    print("Frame computation time:", time.time() - start_time)

    return computed_mandelbrot


def save_frame(complete_mandelbrot, i):
    plt.figure("output", figsize=(pIM, pRE), dpi=1)
    plt.imshow(complete_mandelbrot, cmap='magma', interpolation='nearest', aspect='auto')

    # Formatting of matplotlib figure removing all axes and padding
    plt.axis('off')
    plt.bbox_inches = 'tight'
    plt.pad_inches = 0
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(f'tmp_hold.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    loaded_image = cv2.imread('tmp_hold.png')
    cv2.putText(loaded_image, f'Iterations: {i}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
    video_writer.write(loaded_image)

    # Progress bar and time estimation
    progress_count = round(i / iterations * 100, 2)
    print(f'\n\nProgress: {progress_count}%')
    plt.pause(0.001)  # Pause for 1ms to allow the plot to update


if __name__ == '__main__':
    start_render_time = time.time()

    main(iterations)

    # Hold the last frame for 3 seconds when the video ends
    for i in range(frame_rate*3):
        video_writer.write(cv2.imread('tmp_hold.png'))
    video_writer.release()
    os.remove('tmp_hold.png')  # Remove the temporary image file
    print("Video saved to:", os.getcwd() + "/" + output_video_destination)
    print("Complete! Total render time:", time.time() - start_render_time)