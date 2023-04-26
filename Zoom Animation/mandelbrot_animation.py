from matplotlib import pyplot as plt
import time
import numpy
import cv2
import os
import datetime
import mandelbrot_multicore
import multiprocessing
import video_creator


pRE = 1000
pIM = 1000
threshold = 2
iterations = 1000

frame_rate = 60
step_size = frame_rate*61


def compute_set(x_0, y_0, x_1, y_1):
    print("X:",x_0, x_1, "Y:", y_0, y_1, "")

    start_time = time.time()

    reshaped_solution = mandelbrot_multicore.main(chunk_size=100, cores=multiprocessing.cpu_count(), pRE=pRE, pIM=pIM,
                                                  show_figure=False, x0=x_0, x1=x_1, y0=y_0, y1=y_1)

    print("Frame computation time:", time.time() - start_time)

    return reshaped_solution


def main():
    output_video_destination = f'mandelbrot_animation_{str(datetime.datetime.now()).replace(":", "-").replace(".", "")}.avi'
    print(f'Video save destination: {os.getcwd()}/{output_video_destination}')
    start_render_time = time.time()

    start_point = [-2.25, 0.75, -1.25, 1.25]
    end_point = [-0.7336438924199521-(4.5E-14)/2, -0.7336438924199521+(4.5E-14)/2, 0.2455211406714035-(4.5E-14)/2, 0.2455211406714035+(4.5E-14)/2]

    video_writer = video_creator.VideoCreator(output_video_destination, frame_rate, pRE, pIM)

    x0, x1, y0, y1 = start_point
    a0, a1, b0, b1 = end_point

    for i in range(step_size):
        t = 1-(1-i / step_size)**10

        current_x = [x0 + t * (a0 - x0), x1 + t * (a1 - x1)]
        current_y = [y0 + t * (b0 - y0), y1 + t * (b1 - y1)]

        # Apply the Mandelbrot formula and generate the image
        complete_mandelbrot = compute_set(current_x[0], current_y[0], current_x[1], current_y[1])

        video_writer.create_frame(complete_mandelbrot, [f'X: {round(current_x[0], 4)} : {round(current_x[1], 4)}',
                                                        f'Y: {round(current_y[0], 4)} : {round(current_y[1], 4)}'])

        # Progress bar and time estimation
        progress_count = round((i / step_size) * 100, 2)
        print(f'\n\nProgress: {progress_count}%')
        if i != 0:
            print("Remaining time:", round((time.time() - start_render_time) * (1 - (i / step_size)) / (i / step_size), 2), "seconds")
        plt.pause(0.001)  # Pause for 1ms to allow the plot to update

    # Hold the last frame for 3 seconds when the video ends
    for i in range(frame_rate*3):
        video_writer.save_frame(cv2.imread('tmp_hold.png'))

    video_writer.close()
    os.remove('tmp_hold.png')  # Remove the temporary image file

    print("Video saved to:", os.getcwd() + "/" + output_video_destination)
    print("Complete! Total render time:", time.time() - start_render_time)

if __name__ == "__main__":
    main()