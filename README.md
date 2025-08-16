# CUDA and Multiprocessing Project

This project contains two primary modules: a CUDA-based solver for differential equations and a suite of tools for parallel data processing using Python's multiprocessing library.

## Table of Contents

- [CUDA Differential Equation Solver](#cuda-differential-equation-solver)
  - [Overview](#overview)
  - [Files](#files)
  - [How to Run](#how-to-run)
- [Multiprocessing for Datasets](#multiprocessing-for-datasets)
  - [Overview](#overview-1)
  - [Modules](#modules)

---

## CUDA Differential Equation Solver

### Overview

This module is designed to solve second-order linear ordinary differential equations (ODEs) using the finite difference method, accelerated with NVIDIA CUDA. It demonstrates how to leverage GPU parallelism to speed up the process of matrix inversion and matrix-vector multiplication, which are key components of the solver.

### Files

-   **`matrix_inversion.cu`**: Contains the CUDA C++ kernels for matrix inversion using the Gauss-Jordan elimination method.
-   **`solver.cu`**: The main CUDA C++ file that orchestrates the matrix setup and calls the inversion kernels.
-   **`solver.py`**: The Python script that uses PyCUDA to compile and run the CUDA kernels. It also handles data initialization and result verification.

### How to Run

1.  **Prerequisites**:
    *   NVIDIA GPU with CUDA support.
    *   CUDA Toolkit installed.
    *   Python environment with the necessary packages.

2.  **Execution**:
    Run the Python script from the command line:
    ```bash
    python cuda/differential_solver/solver.py
    ```
    The script will compile the CUDA code, run the solver, and print the results, including the maximum error compared to the analytical solution.

---

## Multiprocessing for Datasets

### Overview

This part of the project provides a collection of Python scripts that use the `concurrent.futures` module to perform common data manipulation tasks in parallel. This is particularly useful for large datasets where performance is a key concern.

### Modules

-   **`aggregation.py`**: Contains functions for performing parallel data aggregation, such as summing up a list of numbers.
-   **`filtering.py`**: Provides functions for filtering data in a pandas DataFrame in parallel based on specified conditions (e.g., by date or value).
-   **`transformation.py`**: Includes functions for applying data transformations, like Min-Max scaling and z-score normalization, in parallel.

---
