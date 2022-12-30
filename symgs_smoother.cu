#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define THREADSPERBLOCKR 1024
#define THREADSPERBLOCKD 64

#define CHECK(call)                                                                       \
    {                                                                                     \
        const cudaError_t err = call;                                                     \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

#define CHECK_KERNELCALL()                                                                \
    {                                                                                     \
        const cudaError_t err = cudaGetLastError();                                       \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

// function to get the time of day in seconds
double get_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Reads a sparse matrix and represents it using CSR (Compressed Sparse Row) format
void read_matrix(int **row_ptr, int **col_ind, float **values, float **matrixDiagonal, const char *filename, int *num_rows, int *num_cols, int *num_vals)
{
    int err;
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        fprintf(stdout, "File cannot be opened!\n");
        exit(0);
    }
    // Get number of rows, columns, and non-zero values
    if(fscanf(file, "%d %d %d\n", num_rows, num_cols, num_vals)==EOF)
        printf("Error reading file");

    int *row_ptr_t = (int *)malloc((*num_rows + 1) * sizeof(int));
    int *col_ind_t = (int *)malloc(*num_vals * sizeof(int));
    float *values_t = (float *)malloc(*num_vals * sizeof(float));
    float *matrixDiagonal_t = (float *)malloc(*num_rows * sizeof(float));
    // Collect occurances of each row for determining the indices of row_ptr
    int *row_occurances = (int *)malloc(*num_rows * sizeof(int));
    for (int i = 0; i < *num_rows; i++)
    {
        row_occurances[i] = 0;
    }

    int row, column;
    float value;
    while (fscanf(file, "%d %d %f\n", &row, &column, &value) != EOF)
    {
        // Subtract 1 from row and column indices to match C format
        row--;
        column--;
        row_occurances[row]++;
    }

    // Set row_ptr
    int index = 0;
    for (int i = 0; i < *num_rows; i++)
    {
        row_ptr_t[i] = index;
        index += row_occurances[i];
    }
    row_ptr_t[*num_rows] = *num_vals;
    free(row_occurances);

    // Set the file position to the beginning of the file
    rewind(file);

    // Read the file again, save column indices and values
    for (int i = 0; i < *num_vals; i++)
    {
        col_ind_t[i] = -1;
    }

    if(fscanf(file, "%d %d %d\n", num_rows, num_cols, num_vals)==EOF)
        printf("Error reading file");

    int i = 0, j = 0;
    while (fscanf(file, "%d %d %f\n", &row, &column, &value) != EOF)
    {
        row--;
        column--;

        // Find the correct index (i + row_ptr_t[row]) using both row information and an index i
        while (col_ind_t[i + row_ptr_t[row]] != -1)
        {
            i++;
        }
        col_ind_t[i + row_ptr_t[row]] = column;
        values_t[i + row_ptr_t[row]] = value;
        if (row == column)
        {
            matrixDiagonal_t[j] = value;
            j++;
        }
        i = 0;
    }
    fclose(file);
    *row_ptr = row_ptr_t;
    *col_ind = col_ind_t;
    *values = values_t;
    *matrixDiagonal = matrixDiagonal_t;
}

// CPU implementation of SYMGS using CSR, DO NOT CHANGE THIS
void symgs_csr_sw(const int *row_ptr, const int *col_ind, const float *values, const int num_rows, float *x, float *matrixDiagonal)
{

    // forward sweep
    for (int i = 0; i < num_rows; i++)
    {
        float sum = x[i];
        const int row_start = row_ptr[i];
        const int row_end = row_ptr[i + 1];
        float currentDiagonal = matrixDiagonal[i]; // Current diagonal value

        for (int j = row_start; j < row_end; j++)
        {
            sum -= values[j] * x[col_ind[j]];
        }

        sum += x[i] * currentDiagonal; // Remove diagonal contribution from previous loop

        x[i] = sum / currentDiagonal;
    }

    // backward sweep
    for (int i = num_rows - 1; i >= 0; i--)
    {
        float sum = x[i];
        const int row_start = row_ptr[i];
        const int row_end = row_ptr[i + 1];
        float currentDiagonal = matrixDiagonal[i]; // Current diagonal value

        for (int j = row_start; j < row_end; j++)
        {
            sum -= values[j] * x[col_ind[j]];
        }
        sum += x[i] * currentDiagonal; // Remove diagonal contribution from previous loop

        x[i] = sum / currentDiagonal;
    }
}

// GPU implementation

__global__ void symgs_csr_hw_fready(int* row_ptr, int* col_ind, int* ready, int* done, int num_rows){
	
	const int threadID = threadIdx.x + blockDim.x * blockIdx.x;
	
	if(threadID < num_rows){
		if(!done[threadID]){
			int rowl = row_ptr[threadID + 1] - row_ptr[threadID];
			
			ready[threadID] = 1;
			for(int j = 0; j < rowl && ready[threadID] && col_ind[row_ptr[threadID] + j] >= 0; j++){
				if(!done[col_ind[row_ptr[threadID] + j]] && col_ind[row_ptr[threadID] + j] < threadID){
					ready[threadID] = 0;
				}
			}
		}
	}
}

__global__ void symgs_csr_hw_bready(int* row_ptr, int* col_ind, int* ready, int* done, int num_rows){
	
	const int threadID = threadIdx.x + blockDim.x * blockIdx.x;
	
	if(threadID < num_rows){
		if(!done[threadID]){
			int rowl = row_ptr[threadID + 1] - row_ptr[threadID];
			
			ready[threadID] = 1;
			for(int j = 0; j < rowl && ready[threadID] && col_ind[row_ptr[threadID] + j] >= 0; j++){
				if(!done[col_ind[row_ptr[threadID] + j]] && col_ind[row_ptr[threadID] + j] > threadID){
					ready[threadID] = 0;
				}
			}
		}
	}
}

__global__ void symgs_csr_hw_fdo(int* row_ptr, int* col_ind, float* values, float* tmp, float* x, float* matrixDiagonal, int* ready, int* done, int* c){
	
	int row = blockIdx.x;
	
	__shared__ float sum;
	__shared__ float x_tmp;
	
	if(!done[row] && ready[row]){
		
		int rowl = row_ptr[row + 1] - row_ptr[row];
		if(threadIdx.x == 0){
			sum = 0;
		} 
		__syncthreads();
		if(threadIdx.x < rowl && col_ind[row_ptr[row] + threadIdx.x] >= 0){
			if(col_ind[row_ptr[row] + threadIdx.x] < row){
				atomicAdd(&sum, values[row_ptr[row] + threadIdx.x] * tmp[col_ind[row_ptr[row] + threadIdx.x]]);
			} else{
				atomicAdd(&sum, values[row_ptr[row] + threadIdx.x] * x[col_ind[row_ptr[row] + threadIdx.x]]);
			}
		}
		__syncthreads();
		if(threadIdx.x == 0){
			x_tmp = x[row] - sum;
			x_tmp += matrixDiagonal[row] * x[row];
			tmp[row] = x_tmp / matrixDiagonal[row];
			done[row] = 1;
			atomicAdd(c, 1);
		}
   	 }
}

__global__ void symgs_csr_hw_bdo(int* row_ptr, int* col_ind, float* values, float* tmp, float* x, float* matrixDiagonal, int* ready, int* done, int* c){
	
	int row = blockIdx.x;
	
	__shared__ float sum;
	__shared__ float x_tmp;
	
	if(!done[row] && ready[row]){
		
		int rowl = row_ptr[row + 1] - row_ptr[row];
		if(threadIdx.x == 0){
			sum = 0;
		} 
		
		if(threadIdx.x < rowl && col_ind[row_ptr[row] + threadIdx.x] >= 0){
			if(col_ind[row_ptr[row] + threadIdx.x] > row){
				atomicAdd(&sum, values[row_ptr[row] + threadIdx.x] * tmp[col_ind[row_ptr[row] + threadIdx.x]]);
			} else{
				atomicAdd(&sum, values[row_ptr[row] + threadIdx.x] * x[col_ind[row_ptr[row] + threadIdx.x]]);
			}
		}
		
		if(threadIdx.x == 0){
			x_tmp = x[row] - sum;
			x_tmp += matrixDiagonal[row] * x[row];
			tmp[row] = x_tmp / matrixDiagonal[row];
			done[row] = 1;
			atomicAdd(c, 1);
		}
	}
}

//main
int main(int argc, const char *argv[])
{

    if (argc != 2)
    {
        printf("Usage: ./exec matrix_file");
        return 0;
    }

    int *row_ptr, *col_ind, num_rows, num_cols, num_vals;
    float *values;
    float *matrixDiagonal;

    const char *filename = argv[1];

    double start_cpu, end_cpu, start_gpu, end_gpu;

    read_matrix(&row_ptr, &col_ind, &values, &matrixDiagonal, filename, &num_rows, &num_cols, &num_vals);
    float *x = (float *)malloc(num_rows * sizeof(float));
    float *x_gpu = (float *)malloc(num_rows * sizeof(float));

    // Generate a random vector
    srand(time(NULL));
    for (int i = 0; i < num_rows; i++)
    {
        x[i] = (rand() % 100) / (rand() % 100 + 1); // the number we use to divide cannot be 0, that's the reason of the +1
        x_gpu[i] = x[i];
    }

    // Compute in sw
    start_cpu = get_time();
    symgs_csr_sw(row_ptr, col_ind, values, num_rows, x, matrixDiagonal);
    end_cpu = get_time();

    // Print time
    printf("SYMGS Time CPU: %.10lf\n", end_cpu - start_cpu);

    //GPU implementation
    int *row_ptr_hw, *col_ind_hw, *ready, *done, *count_hw;
    float *values_hw, *matrixDiagonal_hw, *x_hw, *tmp;

    CHECK(cudaMalloc(&row_ptr_hw, (num_rows + 1) * sizeof(int)));
    CHECK(cudaMalloc(&col_ind_hw, num_vals * sizeof(int)));
    CHECK(cudaMalloc(&values_hw, num_vals * sizeof(float)));
    CHECK(cudaMalloc(&matrixDiagonal_hw, num_rows * sizeof(float)));
    CHECK(cudaMalloc(&x_hw, num_rows * sizeof(float)));
    CHECK(cudaMalloc(&tmp, num_rows * sizeof(float)));
    CHECK(cudaMalloc(&ready, num_rows * sizeof(int)));
    CHECK(cudaMalloc(&done, num_rows * sizeof(int)));
    CHECK(cudaMalloc(&count_hw, sizeof(int)));

    CHECK(cudaMemcpy(row_ptr_hw, row_ptr, (num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(col_ind_hw, col_ind, num_vals * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(values_hw, values, num_vals * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(matrixDiagonal_hw, matrixDiagonal, num_rows * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(x_hw, x_gpu, num_rows * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemset(done, 0, num_rows * sizeof(int)));
    CHECK(cudaMemset(count_hw, 0, sizeof(int)));

    start_gpu = get_time();
    int count = 0;
    //forward sweep
    while(count < num_rows){
	    symgs_csr_hw_fready<<<(num_rows / THREADSPERBLOCKR) + 1, THREADSPERBLOCKR>>>(row_ptr_hw, col_ind_hw, ready, done, num_rows);
	    CHECK_KERNELCALL();
	    CHECK(cudaDeviceSynchronize());
	    symgs_csr_hw_fdo<<<num_rows, THREADSPERBLOCKD>>>(row_ptr_hw, col_ind_hw, values_hw, tmp, x_hw, matrixDiagonal_hw, ready, done, count_hw);
	    CHECK_KERNELCALL();
	    CHECK(cudaDeviceSynchronize());
	    cudaMemcpy(&count, count_hw, sizeof(int), cudaMemcpyDeviceToHost);
    }
    CHECK(cudaMemset(done, 0, num_rows * sizeof(int)));
    CHECK(cudaMemset(count_hw, 0, sizeof(int)));
    count = 0;
    //backward sweep
    while(count < num_rows){
	    symgs_csr_hw_bready<<<(num_rows / THREADSPERBLOCKR) + 1, THREADSPERBLOCKR>>>(row_ptr_hw, col_ind_hw, ready, done, num_rows);
	    CHECK_KERNELCALL();
	    CHECK(cudaDeviceSynchronize());
	    symgs_csr_hw_bdo<<<num_rows, THREADSPERBLOCKD>>>(row_ptr_hw, col_ind_hw, values_hw, x_hw, tmp, matrixDiagonal_hw, ready, done, count_hw);
	    CHECK_KERNELCALL();
	    CHECK(cudaDeviceSynchronize());
	    cudaMemcpy(&count, count_hw, sizeof(int), cudaMemcpyDeviceToHost);
    }
    end_gpu = get_time();
    cudaMemcpy(x_gpu, x_hw, num_rows * sizeof(float), cudaMemcpyDeviceToHost);
    printf("SYMGS Time GPU: %.10lf\n", end_gpu - start_gpu);
	
    float num = 0.0;
    float acc;
    for(int i = 0; i < num_rows; i++){
        if(x_gpu[i] - x[i] > 0.001 || x_gpu[i] - x[i] < -0.001){
			num += 1;
        }
    }
    acc = (num / num_rows) * 100;
	
    printf("%d risultati su %d hanno un errore superiore a 10^-3, precisione del %f%%\n", (int)num, num_rows, 100.0 - acc);

    // Free
    free(row_ptr);
    free(col_ind);
    free(values);
    free(matrixDiagonal);
    free(x);
    free(x_gpu);
    CHECK(cudaFree(row_ptr_hw));
    CHECK(cudaFree(col_ind_hw));
    CHECK(cudaFree(values_hw));
    CHECK(cudaFree(matrixDiagonal_hw));
    CHECK(cudaFree(x_hw));
    CHECK(cudaFree(ready));
    CHECK(cudaFree(done));
    CHECK(cudaFree(tmp));
    CHECK(cudaFree(count_hw));

    return 0;
}
