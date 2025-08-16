import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

mod = SourceModule("""
__global__ void divide_by_diagonal_element(float *matrix, int row, int size)
{
    extern __shared__ float row_elements[];

    int i = row*2*size + row + threadIdx.x;
    int s_i = threadIdx.x;

    row_elements[s_i] = matrix[i];
    __syncthreads();

    row_elements[s_i] /= row_elements[0];
    matrix[i] = row_elements[s_i];
    __syncthreads();
}

__global__ void reduce_column(float *matrix, int col, int size)
{
    __shared__ float submatrix[32][33]; // 33 to avoid bank conflicts

    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int column = blockIdx.x*blockDim.x + threadIdx.x;

    if (row < size){
        int i = row*2*size + column + col;
        submatrix[threadIdx.y][threadIdx.x] = matrix[i];
    }
    __syncthreads();

    if (row != col && row < size) 
    submatrix[threadIdx.y][threadIdx.x] -= 
    matrix[col + row*2*size]*matrix[(col*2*size) + column + col];
    __syncthreads();

    if (row < size){
        int i = row*2*size + column + col;
        matrix[i] = submatrix[threadIdx.y][threadIdx.x];
    }
    __syncthreads();
}

__global__ void copy(float *inverted_matrix, float *matrix, int size)
{
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col =  blockIdx.x*blockDim.x + threadIdx.x;

    if (row < size && col < size){
        inverted_matrix[row*size + col] = matrix[row*2*size + col + size];
    }
}

__global__ void matrix_multiply(float *inverted_matrix, float *b, float *solution, int size)
{
    extern __shared__ float submatrix[];

    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = threadIdx.x;

    if (row < size && col < size){
        submatrix[threadIdx.y*(size+1) + threadIdx.x] = inverted_matrix[row*size + col] * b[col];
    }
    __syncthreads();

    if (col == 0 && row < size) {
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            sum += submatrix[threadIdx.y*(size+1) + i];
        }
        solution[row] = sum;
    }
}
""")

mx = np.int32(97) # number of grid points including end points
n = mx - 2 # number of interior points
xi = np.float32(1.0) # initial x
xf = np.float32(2.0) # final x
fx = xf - xi # the range of x


def p(x):
    """
    Coefficient function for the second derivative term of the ODE.

    Parameters:
    x (float): Independent variable.

    Returns:
    float: The coefficient of the second derivative term.
    """
    return x


def q(x):
    """
    Coefficient function for the first derivative term of the ODE.

    Parameters:
    x (float): Independent variable.

    Returns:
    float: The coefficient of the first derivative term.
    """
    return 1


def r(x):
    """
    Coefficient function for the zero-order term of the ODE.

    Parameters:
    x (float): Independent variable.

    Returns:
    float: The coefficient of the zero-order term.
    """
    return 0


def f(x):
    """
    Non-homogeneous term of the ODE.

    Parameters:
    x (float): Independent variable.

    Returns:
    float: The non-homogeneous term.
    """
    return 0


def y(x):
    """
    Solution of the ODE, if it is known (for testing the accuracy).

    Parameters:
        x (float): Independent variable.

    Returns:
        float: Solution of the ODE.
    """
    return np.log(x)


def initMatrix():
    """
    Initializes and sets up the finite difference augmented matrix A and the non-homogeneous vector b for solving a second-order linear ordinary differential equation (ODE) using the finite difference method.

    The matrix A and vector b are constructed based on the coefficients of the differential equation:
    p(x)d²y/dx² + q(x)dy/dx + r(x)y = f(x), with boundary conditions.

    Returns:
        matrix (numpy.ndarray): A n*2n augmented matrix representing the finite difference approximation of the ODE augmented with the identity matrix for calculating it's inverse.
        b (numpy.ndarray): An n-dimensional vector representing the non-homogeneous part of the ODE.

    Notes:
    - Boundary conditions are incorporated by adjusting the elements of the vector b at the boundaries.
    """
    matrix = np.zeros(2*n*(n+1), dtype=np.float32)
    b = np.zeros(n, dtype=np.float32)

    h = fx/(mx - 1)

    # The central difference coefficients
    p1, p2, p3 = np.float32(1.0), np.float32(-2.0), np.float32(1.0)
    q1, q2, q3 = np.float32(-h/2), np.float32(0.0), np.float32(h/2)
    r1, r2, r3 = np.float32(0.0), np.float32(h*h), np.float32(0.0)

    for i in range(n):
        x = xi + (i+1)*fx/(mx-1)
        matrix[i*2*n + i] = p2*p(x) + q2*q(x) + r2*r(x)
        matrix[i*2*n + (n+i)] = 1.0

        b[i] = f(x)*h*h

        if i == 0:
            matrix[i*2*n + i+1] = p3*p(x) + q3*q(x) + r3*r(x)
            b[i] -= (p1*p(x) + q1*q(x) + r1*r(x))*y(xi)
            continue

        if i == n-1:
            matrix[i*2*n + i-1] = p1*p(x) + q1*q(x) + r1*r(x)
            b[i] -= (p3*p(x) + q3*q(x) + r3*r(x))*y(xf)
            continue

        matrix[i*2*n + i+1] = p3*p(x) + q3*q(x) + r3*r(x)
        matrix[i*2*n + i-1] = p1*p(x) + q1*q(x) + r1*r(x)

    return matrix, b


def checkResult(solution):
    """
    Checks the accuracy of the computed solution against the expected analytical solution of the ODE.

    This function calculates the expected solution at each grid point and compares it to the provided 
    numerical solution. It computes and prints the maximum absolute error and maximum percent error.

    Parameters:
        solution (numpy.ndarray): A 1D array of the computed solution values at each grid point.
    """
    expected_sol = np.zeros(n, dtype=np.float32)

    h = fx/(mx-1)

    maxError = np.float32(0.0)
    maxPercentError = np.float32(0.0)

    for i in range(n):
        x = xi + fx*(i+1)/(mx-1)
        expected_sol[i] = y(x)
        maxError = max(maxError, abs(solution[i] - expected_sol[i]))
        maxPercentError = max(maxPercentError, abs(solution[i] - expected_sol[i])/expected_sol[i])
        print(f"x = {x:.5f}\tCalculated: {solution[i]:.7f}\tExpected: {expected_sol[i]:.7f}")

    print(f"\tMax Error: {maxError}\n")
    print(f"\tMax Percent Error: {maxPercentError}\n")


if __name__ == "__main__":
    matrix, b = initMatrix()

    d_matrix = cuda.mem_alloc(matrix.nbytes)
    cuda.memcpy_htod(d_matrix, matrix)

    gridDim = (int((n + 1)/32), int((n + 1)/32), 1)
    blockDim = (32, 32, 1)

    divide_by_diagonal_element = mod.get_function("divide_by_diagonal_element")
    reduce_column = mod.get_function("reduce_column")

    for i in range(n):
        i = np.int32(i)
        divide_by_diagonal_element(d_matrix, i, n, grid=(1, 1, 1), block=(int(n + 1), 1, 1), shared=int((n + 1)*np.dtype(np.float32).itemsize))
        reduce_column(d_matrix, i, n, grid=gridDim, block=blockDim)

    d_b = cuda.mem_alloc(b.nbytes)
    cuda.memcpy_htod(d_b, b)

    d_inverted_matrix = cuda.mem_alloc(int(n*n*np.dtype(np.float32).itemsize))

    copy = mod.get_function("copy")

    copy(d_inverted_matrix, d_matrix, n, grid=gridDim, block=blockDim)

    solution = np.zeros(n, dtype=np.float32)
    d_solution = cuda.mem_alloc(solution.nbytes)

    matrix_multiply = mod.get_function("matrix_multiply")

    gridDim = (1, int(((n + 1)**2)/1024 + 1), 1)
    blockDim = (int(n + 1), int(1024//(n + 1)), 1)

    matrix_multiply(d_inverted_matrix, d_b, d_solution, n, grid=gridDim, block=blockDim, shared = 1024*np.dtype(np.float32).itemsize)

    cuda.memcpy_dtoh(solution, d_solution)

    checkResult(solution)