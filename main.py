import mandelbrot_naive
import mandelbrot_vectorized
import mandelbrot_numba
import mandelbrot_multicore
import multiprocessing
import mandelbrot_datatypes
import mandelbrot_dask


"""
    Run this file to compare the different implementations of the Mandelbrot set.
    - To show the figures: Re-configure the parameter "show_figure" for each implementation to show the figure or not.
    The default values for the parameters are:
        pRE = 1000
        pIM = 1000
        threshold = 2
        Iterations = 100
"""


def mini_project_part_1():
    print(f'\n {"#"*15} Mini-project Part 1 {"#"*15}')
    print(f'\n {"-"*10} (a) Implemented using naive approach {"-"*10}')
    mandelbrot_naive.main(show_figure=False)
    print(f'\n {"-"*10} (b) Implemented using numpy vectorized {"-"*10}')
    mandelbrot_vectorized.main(1000, 1000, show_figure=False)
    print(f'\n {"-"*10} (c) Implemented using numba {"-"*10}')
    mandelbrot_numba.main(show_figure=False)
    print(f'\n {"-"*10} (d) Implemented using multiprocessing {"-"*10}')
    # Chunk size defined based on best performing computation time (see the report for details)
    mandelbrot_multicore.main(show_figure=False, cores=multiprocessing.cpu_count(), chunk_size=81)


def mini_project_part_2():
    print(f'\n {"#"*15} Mini-project Part 2 {"#"*15}')
    print(f'\n {"-"*10} (a) Implemented using different data-types {"-"*10}')
    mandelbrot_datatypes.main(show_figure=False)
    print(f'\n {"-"*10} (b) Implemented using Dask Array datatype {"-"*10}')
    mandelbrot_dask.main()


if __name__ == '__main__':
    mini_project_part_2()


# todo: Mini-project Part 3

