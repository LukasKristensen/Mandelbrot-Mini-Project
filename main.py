import mandelbrot_naive
import mandelbrot_numpy
import mandelbrot_numba
import mandelbrot_multicore
import multiprocessing

"""
    Run this file to compare the different implementations of the Mandelbrot set.
    - To show the figures: Re-configure the parameter "show_figure" for each implementation to show the figure or not.
    The default values for the parameters are:
        pRE = 1000
        pIM = 1000
        threshold = 2
        Iterations = 100
"""

if __name__ == '__main__':
    print(f'\n {"#"*15} Mini-project Part 1 {"#"*15}')
    print(f'\n {"-"*10} (a) Implemented using naive approach {"-"*10}')
    mandelbrot_naive.main(show_figure=False)
    print(f'\n {"-"*10} (b) Implemented using numpy vectorized {"-"*10}')
    mandelbrot_numpy.main(show_figure=False)
    print(f'\n {"-"*10} (c) Implemented using numba {"-"*10}')
    mandelbrot_numba.main(show_figure=False)
    print(f'\n {"-"*10} (d) Implemented using multiprocessing {"-"*10}')
    # Chunk size defined based on best performing computation time (see the report for details)
    mandelbrot_multicore.main(show_figure=False, cores=multiprocessing.cpu_count(), chunk_size=81)


# todo: Mini-project Part 1
#       [x] (a) Implement using naive approach
#       [x] (b) Implement using numpy vectorized
#       [x] (c) Implement using numba
#       [x] (d) Implement using multiprocessing

# todo: Mini-project Part 2

# todo: Mini-project Part 3

