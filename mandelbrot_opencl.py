"""
    Author: Lukas Bisgaard Kristensen
    Date: 26. April 2023
    Course: Numerical Scientific Computing, AAU
    Description: This program computes the Mandelbrot set using OpenCL.
"""

import pyopencl
import matplotlib.pyplot as plt
import time
import numpy
import doctest
import os
# os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

compute_local_sizes = [[]]
compute_local_computations = [[]]


def mandelbrot_opencl(device, context, queue, x_min=-2.3, x_max=0.8, y_min=-1.2, y_max=1.2, width=5000, height=5000, show_figure=True, local_size=1) -> None:
    """
    Computes the Mandelbrot set using OpenCL.

    :param device: Device name of CPU/GPU
    :param context: Context of the device
    :param queue: Queue of the device
    :param x_min: Minimum real value
    :param x_max: Maximum real value
    :param y_min: Minimum imaginary value
    :param y_max: Maximum imaginary value
    :param width: Width of the image
    :param height: Height of the image
    :param show_figure: Show the figure
    :return: Mandelbrot set

    >>> import pyopencl
    >>> import numpy as np
    >>> from matplotlib import pyplot as plt
    >>> device = pyopencl.get_platforms()[0].get_devices()[0]
    >>> context = pyopencl.Context([device])
    >>> queue = pyopencl.CommandQueue(context, device)
    >>> x_min, x_max, y_min, y_max, width, height = -2.3, 0.8, -1.2, 1.2, 5000, 5000
    >>> mandelbrot_opencl(device, context, queue, x_min, x_max, y_min, y_max, width, height, show_figure=False, local_size=1)
    array([[1., 1., 1., ..., 2., 2., 2.],
           [1., 1., 1., ..., 2., 2., 2.],
           [1., 1., 1., ..., 2., 2., 2.],
           ...,
           [1., 1., 1., ..., 2., 2., 2.],
           [1., 1., 1., ..., 2., 2., 2.],
           [1., 1., 1., ..., 2., 2., 2.]], dtype=float32)
    """

    start_time = time.time()

    x_space = numpy.linspace(x_min, x_max, width, dtype=numpy.float32)
    y_space = numpy.linspace(y_min, y_max, height, dtype=numpy.float32)
    complete_space = (x_space + y_space[:, numpy.newaxis] * 1j).astype(numpy.complex64)

    output = numpy.empty(width * height, dtype=numpy.float32)

    program = pyopencl.Program(context, """
    __kernel void mandelbrot(__global float2 *complete_space, __global float *output)
    {
        __private int gid = get_global_id(0);
        __private float nreal, real = 0;
        __private float imaginary = 0;
        __private float real_pow_2, imaginary_pow_2;
                
        for (int i = 0; i < 3000; i++)
        {
            real_pow_2 = real * real;
            imaginary_pow_2 = imaginary * imaginary;
            
            nreal = real_pow_2 - imaginary_pow_2 + complete_space[gid].x;
            imaginary = 2 * real * imaginary + complete_space[gid].y;
            real = nreal;
            if (real_pow_2 + imaginary_pow_2 > 4)
            {
                output[gid] = i;
                return;
            }
        }
    }
    """).build(devices=[device])

    mf = pyopencl.mem_flags
    q_opencl = pyopencl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=complete_space)
    output_opencl = pyopencl.Buffer(context, mf.WRITE_ONLY, output.nbytes)

    program.mandelbrot(queue, output.shape, [local_size], q_opencl, output_opencl)
    pyopencl.enqueue_copy(queue, output, output_opencl).wait()

    divergence_time = output.reshape((height, width))
    time_compute = computation_time(start_time, time.time())
    print(f"Computation time: {time_compute} seconds")

    compute_local_sizes[-1].append(local_size)
    compute_local_computations[-1].append(time_compute)

    if show_figure:
        plt.imshow(divergence_time, cmap='magma')
        plt.show()
    return divergence_time


def computation_time(start_time, end_time):
    """
    Computes the time taken to compute the Mandelbrot set.

    :param start_time: Start time of computation
    :param end_time: End time of computation
    :return: Difference between the end time and the start time

    Usage examples:
    >>> computation_time(0, 0.792)
    0.792
    """
    return round(end_time - start_time, 3)


def create_opencl_context(platform):
    """
    Create OpenCL context, queue, device and platform

    Parameters
    :param platform: Name of the platform to use
    :return: context, queue, device, name: Output from the CPU/GPU

    Usage examples:
    >>> import pyopencl
    >>> platform = pyopencl.get_platforms()[0]
    >>> context, queue, device, name = create_opencl_context(platform)
    >>> isinstance(context, pyopencl.Context)
    True
    >>> isinstance(queue, pyopencl.CommandQueue)
    True
    >>> isinstance(device, pyopencl.Device)
    True
    >>> isinstance(name, str)
    True
    """

    device = platform.get_devices()[0]
    context = pyopencl.Context(devices=[device])
    queue = pyopencl.CommandQueue(context)
    return context, queue, device, platform.name


def main(show_fig=False):
    """
    Main function for running the comparisons between CPU and GPU and plot sizes.

    :param show_fig: Show the figure or not when finishing the computations
    """
    global compute_local_sizes, compute_local_computations
    compute_local_sizes = []
    compute_local_computations = []
    platform_name = []

    global_sizes = [500, 1000, 2000, 5000, 10000]
    local_sizes = [2, 4, 8, 16, 32, 64, 128, 256]

    for i in pyopencl.get_platforms():
        context, queue, device, name = create_opencl_context(i)
        print("Platform:", name)

        platform_name.append(name)
        compute_local_sizes.append([])
        compute_local_computations.append([])
        for size in local_sizes:
            print("Local size:", size)
            try:
                mandelbrot_opencl(device=device, context=context, queue=queue, width=10000, height=10000, local_size=size, show_figure=False)
            except:
                print("Error in local size:", size)
                break

    for i in range(len(platform_name)):
        plt.plot(max(compute_local_sizes), compute_local_computations[i], label=platform_name[i])

    plt.xlabel("Local Size")
    plt.ylabel("Computation Time (s)")
    plt.title("Computation Time vs Local Size")
    plt.legend()
    plt.show()

    if show_fig:
        context, queue, device, name = create_opencl_context(pyopencl.get_platforms()[0])
        mandelbrot_opencl(device=device, context=context, queue=queue, width=10000, height=10000, local_size=1, show_figure=True)


if __name__ == '__main__':
    doctest.testmod(report=True, verbose=True)
    main(show_fig=True)
