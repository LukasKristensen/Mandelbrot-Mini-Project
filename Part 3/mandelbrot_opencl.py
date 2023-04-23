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


def mandelbrot_opencl(device, context, queue, x_min=-2.3, x_max=0.8, y_min=-1.2, y_max=1.2, width=5000, height=5000, show_figure=True) -> None:
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
    >>> mandelbrot_opencl(device, context, queue, x_min, x_max, y_min, y_max, width, height, show_figure=False)
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
                
        for (int i = 0; i < 100; i++)
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

    program.mandelbrot(queue, output.shape, None, q_opencl, output_opencl)
    pyopencl.enqueue_copy(queue, output, output_opencl).wait()

    divergence_time = output.reshape((height, width))
    time_compute = computation_time(start_time, time.time())
    # print(f"Computation time: {time_compute} seconds")

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


def create_opencl_context(platform_name: str = "Intel"):
    """
    Create OpenCL context, queue, device and platform

    Parameters
    :param platform_name: Name of the platform to use
    :return: cpu_context, cpu_queue, cpu_device, cpu_name: Output from the CPU/GPU

    Usage examples:
    >>> cpu_context, cpu_queue, cpu_device, cpu_name = create_opencl_context()
    >>> assert isinstance(cpu_context, pyopencl.Context)
    >>> assert isinstance(cpu_queue, pyopencl.CommandQueue)
    >>> assert isinstance(cpu_device, pyopencl.Device)
    >>> assert isinstance(cpu_name, str)
    >>> assert cpu_name.lower() in [platform.name.lower() for platform in pyopencl.get_platforms()]
    """

    for i in pyopencl.get_platforms():
        if platform_name in i.name:
            cpu_platform = i
            cpu_device = cpu_platform.get_devices()[0]
            cpu_context = pyopencl.Context(devices=[cpu_device])
            cpu_queue = pyopencl.CommandQueue(cpu_context)
            cpu_name = i.name
            return cpu_context, cpu_queue, cpu_device, cpu_name


def main(show_fig=False):
    """
    Main function for running the comparisons between CPU and GPU and plot sizes.

    :param show_fig: Show the figure or not when finishing the computations
    """

    sizes_to_compute = [500, 1000, 2000, 5000, 10000]

    cpu_context, cpu_queue, cpu_device, cpu_name = create_opencl_context(platform_name="Intel")
    gpu_context, gpu_queue, gpu_device, gpu_name = create_opencl_context(platform_name="NVIDIA")

    if cpu_context:
        print("CPU:", cpu_name)
        for size in sizes_to_compute:
            mandelbrot_opencl(device=cpu_device, context=cpu_context, queue=cpu_queue, width=size, height=size, show_figure=False)
    else:
        print("No CPU found")
    if gpu_context:
        print("GPU:", gpu_name)
        for size in sizes_to_compute:
            mandelbrot_opencl(device=gpu_device, context=gpu_context, queue=gpu_queue, width=size, height=size, show_figure=False)
        mandelbrot_opencl(device=gpu_device, context=gpu_context, queue=gpu_queue, width=10000, height=10000, show_figure=show_fig)
    else:
        print("No GPU found")


if __name__ == '__main__':
    doctest.testmod(report=True, verbose=True)
    main(show_fig=True)
