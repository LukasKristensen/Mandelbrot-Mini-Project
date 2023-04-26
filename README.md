# Mandelbrot-Mini-Project
Implementation of the Mandelbrot set for the Numerical Scientific Computing Course (Computer Engineering 8th semester, AAU)

## Features
- Compare the different algorithm approaches: [``main.py``](main.py)
- Mandelbrot Interactive Navigator [``mandelbrot_navigator.py``](mandelbrot_navigator.py)
- Mandelbrot Zoom Animation [``Zoom Animation/mandelbrot_animation.py``](<Zoom Animation/mandelbrot_animation.py>)
  - Video: https://www.youtube.com/watch?v=L2zKIrriDfI
- Mandelbrot Iteration Animation [``Iterations Animation/mandelbrot_iteration_animation.py``](<Iterations Animation/mandelbrot_iteration_animation.py>)
  - Video: https://www.youtube.com/watch?v=8BjqgaIuses

</br>

<!----------------------------------------->

## Generated Content

### High resolution render
High resoultion render of the mandelbrot set [``HighResolutionRender/ImageRender.py``](<High Resolution Image Render/ImageRender.py>)

![img](<High Resolution Image Render/MandelbrotOutput.png>)
**Image computed using:**
- Size: (1e4, 1e4)
- Iterations: 100

</br>

### Animated Zoom

Generated sequence of zoom into fractal [``mandelbrot_animation.py``](<Zoom Animation/mandelbrot_animation.py>)

[Demo of zooming into fractal](https://www.youtube.com/watch?v=L2zKIrriDfI)

</br>

### Animated Iterations

Generated sequence of zoom into fractal [``mandelbrot_iteration_animation.py``](<Iterations Animation/mandelbrot_iteration_animation.py>)

[Demo of animating iterations](https://www.youtube.com/watch?v=8BjqgaIuses)

</br>

## Interactive Navigator

Navigation through the Mandelbrot Set by using keyboard controls [``mandelbrot_navigator.py``](mandelbrot_navigator.py)

![img](<interactive_screenshot.png>)

</br>


<!----------------------------------------->

## Performance Results
| Approach    | Computation Time (s) |
| ----------- | ----------- |
| [``mandelbrot_naive.py``](mandelbrot_naive.py)| 393.58 |
| [``mandelbrot_vectorized.py``](mandelbrot_vectorized.py)| 160.92 |
| [``mandelbrot_numba.py``](mandelbrot_numba.py)| 115.25 |
| [``mandelbrot_multicore.py``](mandelbrot_multicore.py)| 37.89 |
| [``mandelbrot_dask.py``](mandelbrot_dask.py)| 25.07 |
| [``mandelbrot_opencl.py``*](mandelbrot_opencl.py)| 1.07 |

*Limited to float64 precision


|  | Complex64 | Complex128 |
| ----------- | ----------- | ----------- |
| np.float16 | 155.10s | 193.18s |
| np.float32 | 157.76s | 199.54s |
| np.float64 | 199.08s | 223.37s |


**Parameters**
- Size: 10.000x10.000
- Iterations: 100

**Specs**
- Intel® Core™ i5-11300H-processor @ 3.10GHz
- 16 GB DDR4 Ram @ 3200MHz

</br>

<!----------------------------------------->

## Setup
Install the necessary packages by running the command within the root project directory:

```shell
pip install -r requirements.txt
```
</br>


<!----------------------------------------->

## Group
- Lukas Bisgaard Kristensen 
