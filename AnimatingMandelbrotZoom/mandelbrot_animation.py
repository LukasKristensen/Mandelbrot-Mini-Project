from matplotlib import pyplot as plt
import time
import numpy
import numba
import cv2
import os
import datetime

pRE = 500
pIM = 500
threshold = 2

frames_between_points = 100
frame_rate = round(frames_between_points/6)

output_video_destination = f'mandelbrot_animation_{str(datetime.datetime.now()).replace(":", "-")}.avi'
print(f'Video save destination: {os.getcwd()}/{output_video_destination}')


def mandelbrot(c):
    """
    Generate a Mandelbrot set using vectorized numpy operations.

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
    for i in range(100):
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
    x_space = numpy.linspace(x_0, x_1, pRE, dtype=numpy.float32).reshape((1, pRE))
    y_space = numpy.linspace(y_0, y_1, pIM, dtype=numpy.float32).reshape((pIM, 1))

    # Generate a 2D array for each dimension of the complex plane
    complete_space = x_space + y_space * 1j

    # Apply the Mandelbrot formula
    computed_mandelbrot = mandelbrot(complete_space)

    print("Frame computation time:", time.time() - start_time)

    return computed_mandelbrot


if __name__ == '__main__':
    start_render_time = time.time()

    # Coordinates to interpolate between
    # First four coordinates are the start and end points of the interpolation
    # The fifth coordinate is the number of steps to take between the start and end points
    interest_points = [[-2.25, 0.75, -1.25, 1.25, frames_between_points],
                       [-0.352917, -0.127973, -0.722195, -0.534797, frames_between_points],
                       [-0.206791, -0.195601, -0.68154, -0.672274, frames_between_points],
                       [-0.19925116, -0.199134805, -0.679549605, -0.67945249, frames_between_points]]

    video_writer = cv2.VideoWriter(output_video_destination, cv2.VideoWriter_fourcc(*'DIVX'), frame_rate, (pRE, pIM))

    for z in range(1, len(interest_points)):
        start_x, start_y = (interest_points[z-1][0], interest_points[z-1][1]), (interest_points[z-1][2], interest_points[z-1][3])
        target_x = (interest_points[z][0], interest_points[z][1])
        target_y = (interest_points[z][2], interest_points[z][3])
        step_size = interest_points[z][4]

        dif_x, dif_y = (target_x[0]-start_x[0], target_x[1]-start_x[1]), (target_y[0]-start_y[0], target_y[1]-start_y[1])
        step_x, step_y = (dif_x[0]/step_size, dif_x[1]/step_size), (dif_y[0]/step_size, dif_y[1]/step_size)

        print("Differences:", dif_x, dif_y)
        print("Steps:", step_x, step_y)
        print("\n\n")

        for i in range(step_size+1):
            print("Applying step calculations:",step_x[0]*i, step_x[1]*i, step_y[0]*i, step_y[1]*i)
            current_x = (start_x[0]+step_x[0]*i, start_x[1]+step_x[1]*i)
            current_y = (start_y[0]+step_y[0]*i, start_y[1]+step_y[1]*i)

            # Apply the Mandelbrot formula and generate the image
            complete_mandelbrot = main(current_x[0], current_y[0], current_x[1], current_y[1])
            plt.imshow(complete_mandelbrot, cmap='magma')
            plt.figure("output", figsize=(pIM, pRE), dpi=1)

            # Formatting of matplotlib figure removing all axes and padding
            plt.axis('off')
            plt.bbox_inches = 'tight'
            plt.pad_inches = 0
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(f'tmp_hold.png', bbox_inches='tight', pad_inches=0)
            video_writer.write(cv2.imread('tmp_hold.png'))

            # Progress bar and time estimation
            progress_count = round(((i+((z-1)*step_size))/(len(interest_points)-1)/step_size)*100,2)
            print(f'\n\nProgress: {round(((i+((z-1)*step_size))/(len(interest_points)-1)/step_size)*100,2)}%')
            if progress_count != 0:
                time_est = str(datetime.timedelta(seconds=((time.time() - start_render_time)/progress_count) *
                                                          (100-progress_count))).split(":")
                print(f'Estimated to finish in {time_est[0]}h {time_est[1]}m {time_est[2].split(".")[0]}s')
            plt.pause(0.001)  # Pause for 1ms to allow the plot to update

    # Hold the last frame for 3 seconds when the video ends
    for i in range(frame_rate*3):
        video_writer.write(cv2.imread('tmp_hold.png'))
    video_writer.release()
    os.remove('tmp_hold.png')  # Remove the temporary image file
    print("Video saved to:", os.getcwd() + "/" + output_video_destination)
    print("Complete! Total render time:", time.time() - start_render_time)
