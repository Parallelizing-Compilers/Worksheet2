# Worksheet 2 README

Optimize convolution code and benchmark its performance against a baseline implementation!

## Files
- `conv_baseline.c`: Baseline convolution implementation.
- `conv_optimized.c`: Your optimized convolution implementation.
- `benchmark.py`: Benchmarking script to compare performance of baseline and optimized implementations.

## Installation
- If you have a version of python with numpy and scipy, you can skip this step. Otherwise, you can install Poetry for Python dependency management at https://python-poetry.org/docs/#installation, and then install required Python packages using Poetry:
    ```bash
    poetry install --no-root
    ```
- Ensure you have a C compiler (e.g., `gcc`) installed to compile the benchmarking harness.

- Build the C benchmarking harness by running `make` in the terminal.

## Instructions

1. Run the benchmark script to compare the performance of the baseline and optimized convolution implementations:

   ```bash
   poetry run python3 benchmark.py
   ```

   (You can leave off the `poetry run` if you have the dependencies installed in your global Python environment.)

    What p-value do you get? Is there a significant difference between the two implementations?

2. Optimize the convolution implementation in `conv_optimized.c` to improve performance. Some ideas for optimization include:
    - Padding: Avoid boundary checks by padding the input array.
    - Loop unrolling: Reduce loop overhead by unrolling the inner loops.
    - Memory access patterns: Ensure data is accessed in a cache-friendly manner.
    - Common subexpression elimination: Reduce redundant calculations within the convolution loops by computing the x offsets into a temporary array, then using that to compute the y offsets.
    - Parallelization: Use multi-threading (e.g., OpenMP) to parallelize the convolution operation across multiple CPU cores.
What optimizations did you implement?

3. After making optimizations, re-run the benchmark script to evaluate performance improvements. What's the best speedup you can achieve?