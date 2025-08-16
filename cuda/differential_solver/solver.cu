#include <stdio.h>
#include "matrix_inversion.cu"

const int mx = 65;
const float fx = 12.0f;
const int n = mx - 2; // n+1 should be a multiple of 32
const float a0 = 0.0f; // value of f at the first point
const float an = 288.0f; // value of f at the last point

void initMatrix(float *matrix_data, float **matrix)
{
    float c1 = 1.0f, c2 = -2.0f, c3 = 1.0f;

    for (int i = 0; i < n; i++){
        matrix[i] = &matrix_data[2*i*n];
    }

    memset(matrix_data, 0, 2*n*(n+1)*sizeof(float));

    for (int i = 0; i < n; i++){
        matrix[i][i] = c2;
        matrix[i][n+i] = 1.0f; // augmented matrix
        if (i != 0) matrix[i][i-1] = c1;
        if (i != n-1) matrix[i][i+1] = c3;
    }
}

// d2y/dx2 = x
// for inner points
void initInput(float *b)
{
    float h = fx/(mx - 1);

    for (int i = 0; i < n; i++){
        b[i] = fx*(i + 1)/(mx - 1)*h*h;
        if (i == 0) b[i] -= a0;
        if (i == (n - 1)) b[i] -= an;
    }
}

void checkResult(float *solution)
{
    float *expectedSol = new float[n];
    float maxError = 0.0f;

    for (int i = 0; i < n; i++){
        float x = fx*(i + 1)/(mx - 1);
        expectedSol[i] = x*x*x/6;
        maxError = fmax(maxError, fabs(solution[i] - expectedSol[i]));
        printf("x = %f\tCalculated: %f\tExpected: %f\n", fx*(i + 1)/(mx - 1), solution[i], expectedSol[i]);
    }

    printf("\nMax Error: %f\n", maxError);

    delete[] expectedSol;
}

__global__ void copy(float *inverted_matrix, float *matrix, int size)
{
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col =  blockIdx.x*blockDim.x + threadIdx.x;

    if (row < n && col < n){
        inverted_matrix[row*size + col] = matrix[row*2*size + col + n];
    }
}

__global__ void matrix_multiply(float *inverted_matrix, float *b, float *solution, int size)
{
    __shared__ float submatrix[1024/(n+1)][n+2];

    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = threadIdx.x;

    if (row < n && col < n){
        submatrix[threadIdx.y][threadIdx.x] = inverted_matrix[row*size + col]*b[col];
    }
    __syncthreads();

    if (col == 0 && row < n) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += submatrix[threadIdx.y][i];
        }
        solution[row] = sum;
    }
    __syncthreads();

    // if (col == 0 && row < n){
    //     float sum = 0.0f;
    //     for (int i = 0; i < n; i++){
    //         sum += inverted_matrix[row*size + i]*b[i];
    //     }
    //     solution[row] = sum;
    // }
}

int main()
{
    float *matrix_data, *d_matrix_data;
    float **matrix;

    cudaMallocHost((void**)&matrix_data, 2*n*(n+1)*sizeof(float));
    // n+1 to avoid accessing non-allocated memory
    // later in allocation of shared memory it accesses the last row also
    cudaMallocHost((void**)&matrix, n*sizeof(float*));

    initMatrix(matrix_data, matrix);

    dim3 gridDim((n+1)/32, (n+1)/32);
    dim3 blockDim(32, 32);
    float milliseconds;
    cudaEvent_t start_event, stop_event;
    cudaEventCreate(&start_event);
    cudaEventCreate(&stop_event);

    cudaEventRecord(start_event);
    cudaMalloc((void**)&d_matrix_data, 2*n*(n+1)*sizeof(float));
    cudaMemcpy(d_matrix_data, matrix_data, 2*n*(n+1)*sizeof(float), cudaMemcpyHostToDevice);

    for (int i = 0; i < n; i++){
        divide_by_diagonal_element<<<1, n+1, (n+1)*sizeof(float)>>>(d_matrix_data, i, n);
        reduce_column<<<gridDim, blockDim>>>(d_matrix_data, i, n);
    }

    cudaMemcpy(matrix_data, d_matrix_data, 2*n*(n+1)*sizeof(float), cudaMemcpyDeviceToHost);


    float *b, *d_b;

    cudaMallocHost((void**)&b, n*sizeof(float));

    initInput(b);

    cudaMalloc((void**)&d_b, n*sizeof(float));
    cudaMemcpy(d_b, b, n*sizeof(float), cudaMemcpyHostToDevice);


    float *inverted_matrix, *d_inverted_matrix;

    cudaMalloc((void**)&d_inverted_matrix, n*n*sizeof(float));
    copy<<<gridDim, blockDim>>>(d_inverted_matrix, d_matrix_data, n);

    cudaMallocHost((void**)&inverted_matrix, n*n*sizeof(float));
    cudaMemcpy(inverted_matrix, d_inverted_matrix, n*n*sizeof(float), cudaMemcpyDeviceToHost);


    float *solution, *d_solution;

    dim3 gridDim1(1, ((n+1)*(n+1))/1024 + 1);
    dim3 blockDim1((n+1), 1024/(n+1));

    cudaMalloc((void**)&d_solution, n*sizeof(float));
    matrix_multiply<<<gridDim1, blockDim1>>>(d_inverted_matrix, d_b, d_solution, n);

    cudaMallocHost(&solution, n*sizeof(float));
    memset(solution, 0, n*sizeof(float));
    cudaMemcpy(solution, d_solution, n*sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    cudaEventElapsedTime(&milliseconds, start_event, stop_event);

    checkResult(solution);
    printf("\n\tTime Taken (ms): %f\n", milliseconds);

    cudaFreeHost(matrix_data);
    cudaFreeHost(matrix);
    cudaFreeHost(b);
    cudaFreeHost(inverted_matrix);
    cudaFreeHost(solution);

    cudaFree(d_matrix_data);
    cudaFree(d_b);
    cudaFree(d_inverted_matrix);
    cudaFree(d_solution);
}