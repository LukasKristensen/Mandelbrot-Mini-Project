import mandelbrot_naive
import mandelbrot_vectorized
import mandelbrot_numba
import mandelbrot_multicore
import multiprocessing
import mandelbrot_datatypes
import mandelbrot_dask
import mandelbrot_opencl
import pyopencl
import mandelbrot_navigator


def mini_project_part_1():
    print(f'\n {"#"*15} Mini-project Part 1 {"#"*15}')
    print(f'\n {"-"*10} (a) Implemented using naive approach {"-"*10}')
    mandelbrot_naive.main(1000, 1000, show_figure=False)
    print(f'\n {"-"*10} (b) Implemented using numpy vectorized {"-"*10}')
    mandelbrot_vectorized.main(1000, 1000, show_figure=False)
    print(f'\n {"-"*10} (c) Implemented using numba {"-"*10}')
    mandelbrot_numba.main(1000, 1000, show_figure=False)
    print(f'\n {"-"*10} (d) Implemented using multiprocessing {"-"*10}')
    # Chunk size defined based on best performing computation time (see the report for details)
    mandelbrot_multicore.main(show_figure=False, cores=multiprocessing.cpu_count(), chunk_size=81)


def mini_project_part_2():
    print(f'\n {"#"*15} Mini-project Part 2 {"#"*15}')
    print(f'\n {"-"*10} (a) Implemented using different data-types {"-"*10}')
    mandelbrot_datatypes.main(show_figure=False)
    print(f'\n {"-"*10} (b) Implemented using Dask Array datatype {"-"*10}')
    mandelbrot_dask.main()


def performance_compare(size):
    print(f'\n{"#"*15} Performance Comparison {"#"*15}')
    print("Parameters: size =", size, "x", size, "pixels")

    print("\nNaive approach:")
    mandelbrot_naive.main(show_figure=False, pRE=size, pIM=size)

    print("\nVectorized approach:")
    mandelbrot_vectorized.main(show_figure=False, pRE=size, pIM=size)

    print("\nNumba approach:")
    mandelbrot_numba.main(show_figure=False, pRE=size, pIM=size)

    print("\nMultiprocessing approach:")
    mandelbrot_multicore.main(show_figure=False, cores=multiprocessing.cpu_count(), chunk_size=500, pRE=size, pIM=size)

    print("\nDask approach:")
    mandelbrot_dask.dask_local_distribution(show_figure=False, pRE=size, pIM=size, chunk_size=2000)

    print("\nOpenCL approach:")
    context, queue, device, name = mandelbrot_opencl.create_opencl_context(pyopencl.get_platforms()[0])
    mandelbrot_opencl.mandelbrot_opencl(device=device, context=context, queue=queue, width=size, height=size, local_size=25, show_figure=False)


def mandelbrot_explore():
    mandelbrot_navigator.main()


if __name__ == '__main__':
    # performance_compare(size=10000)
    print(f'\n {"#"*15} Mandelbrot Set {"#"*15}')
    print(f'1) Performance Comparison')
    print(f'2) Mandelbrot Navigation Tool')

    choice = input("Enter your choice: ")
    if choice == '1':
        plot_size = input("\nEnter the size of the plot (e.g. 1000): ")
        performance_compare(size=int(plot_size))
    elif choice == '2':
        mandelbrot_explore()
    else:
        print("Invalid choice!")


