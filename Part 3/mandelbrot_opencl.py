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

cpu_context, cpu_queue, cpu_device, cpu_name = None, None, None, None
gpu_context, gpu_queue, gpu_device, gpu_name = None, None, None, None

for i in pyopencl.get_platforms():
    if "Intel" in i.name:
        print("CPU platform found")
        cpu_platform = i
        cpu_device = cpu_platform.get_devices()[0]
        cpu_context = pyopencl.create_some_context()
        cpu_queue = pyopencl.CommandQueue(cpu_context)
        cpu_name = i.name
    elif "NVIDIA" in i.name:
        gpu_platform = i
        gpu_device = gpu_platform.get_devices()[0]
        gpu_context = pyopencl.create_some_context()
        gpu_queue = pyopencl.CommandQueue(gpu_context)
        gpu_name = i.name


def mandelbrot_opencl(context, queue, x_min=-2.3, x_max=0.8, y_min=-1.2, y_max=1.2, width=5000, height=5000, show_figure=True) -> None:
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
    """).build()

    mf = pyopencl.mem_flags
    q_opencl = pyopencl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=complete_space)
    output_opencl = pyopencl.Buffer(context, mf.WRITE_ONLY, output.nbytes)

    program.mandelbrot(queue, output.shape, None, q_opencl, output_opencl)
    pyopencl.enqueue_copy(queue, output, output_opencl).wait()

    divergence_time = output.reshape((height, width))
    print(f"Size: {width} OpenCL: {round(time.time() - start_time, 4)} seconds")

    if show_figure:
        plt.imshow(divergence_time, cmap='magma')
        plt.show()


if __name__ == '__main__':
    doctest.testmod(report=True, verbose=True)

    print("CPU:", cpu_name)
    if cpu_context:
        for i in [500, 1000, 2000, 5000, 10000]:
            mandelbrot_opencl(context=cpu_context, queue=cpu_queue, width=i, height=i, show_figure=False)
    print("GPU:", gpu_name)
    if gpu_context:
        for i in [500, 1000, 2000, 5000, 10000]:
            mandelbrot_opencl(context=gpu_context, queue=gpu_queue, width=i, height=i, show_figure=False)
    mandelbrot_opencl(context=cpu_context, queue=cpu_queue, width=10000, height=10000, show_figure=True)

