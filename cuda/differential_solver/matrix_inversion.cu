#include <stdio.h>

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