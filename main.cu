
#include <cuda_runtime.h>
#include <mma.h>
#include <random>
#include <iostream>
#include <cmath>
#include <cstdio>
#include <cub/cub.cuh>
using namespace nvcuda;
#define TYPE_NAME int
#define TYPE_NAME2 uint8_t
#define BLOCK_SIZE 256
#define BLOCK_NUM 2000
#define WARPSIZE 32
#define TCSIZE 16
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define TCSIZE_INT8 16
#define WMMA_M_INT 16
#define WMMA_N_INT 16
#define WMMA_K_INT 16

static void HandleError(cudaError_t err, const char* file, int line) { 
	if (err != cudaSuccess) { printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);        
	exit(EXIT_FAILURE); 
	}
}
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

__global__ void convertFp32ToFp16(half* out, float* in, int n) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < n) {
		out[idx] = __float2half(in[idx]);
	}
}
__global__ void convertFp16ToFp32(float* out, half* in, int n) {
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx < n) {
		out[idx] = in[idx];
	}
}

template<typename T>
void initial(T* inputs, int num) {
	srand(time(NULL));
	for (int i = 0; i < num; i++) {
		inputs[i] = rand() % 256;
	}
}

template<typename T>
int cpu_sum_int(T* inputs, int num) {
	int sum = 0;
	for (int i = 0; i < num; i++) {
		sum += inputs[i];
	}
	return sum;
}
/// <summary>
template<typename T>
__inline__ __device__ T warp_sum_utils(T val) {
	val += __shfl_xor_sync(0xffffffff, val, 16);
	val += __shfl_xor_sync(0xffffffff, val, 8);
	val += __shfl_xor_sync(0xffffffff, val, 4);
	val += __shfl_xor_sync(0xffffffff, val, 2);
	val += __shfl_xor_sync(0xffffffff, val, 1);
	return val;
}

template<typename T>
__inline__ __device__ T warp_block_sum(T val) {
	__shared__ T s_data[32];
	int warpIdx = threadIdx.x / 32;
	int laneIdx = threadIdx.x % 32;
	val = warp_sum_utils(val);
	if (laneIdx == 0) {
		
		s_data[warpIdx] = val;
	}
	__syncthreads();
	val = (threadIdx.x < (blockDim.x / 32)) ? s_data[threadIdx.x] : 0;
	if (warpIdx == 0) {
		val += __shfl_down_sync(0xffffffff, val, 4);
		val += __shfl_down_sync(0xffffffff, val, 2);
		val += __shfl_down_sync(0xffffffff, val, 1);//block is 256
	}
	return val;
}

__inline__ __device__ int block_reduce_tc_int_v2(uint8_t* a, int N, int offset) {
	
		int lane = threadIdx.x & (WARPSIZE - 1);
		int wid = threadIdx.x >> 5;
		wmma::fragment<wmma::matrix_a, WMMA_M_INT, WMMA_N_INT, WMMA_K_INT, uint8_t, wmma::row_major> a_frag;
		wmma::fragment<wmma::matrix_b, WMMA_M_INT, WMMA_N_INT, WMMA_K_INT, uint8_t, wmma::col_major> b_frag;
		wmma::fragment<wmma::accumulator, WMMA_M_INT, WMMA_N_INT, WMMA_K_INT, int> d_frag;

		// (1) cargar datos de memoria global a A, B y C frags
		wmma::fill_fragment(a_frag, 1);
		//wmma::fill_fragment(b_frag, 0.0f);
		wmma::fill_fragment(d_frag, 0);
		///
		wmma::load_matrix_sync(b_frag, a + offset + wid * WARPSIZE, TCSIZE_INT8);
		wmma::mma_sync(d_frag, a_frag, b_frag, d_frag);

		__shared__ int shared[8];
		//store_matrix_sync(shared, d_frag, 16, wmma::mem_row_major);

		int val = d_frag.x[0] +d_frag.x[1];
		if (lane == 0) {
			shared[wid] = val;
		}
		__syncthreads();
			val = threadIdx.x < 8 ? shared[threadIdx.x] : 0;
		if (wid == 0) {
			val += __shfl_down_sync(0xffffffff, val, 4);
			val += __shfl_down_sync(0xffffffff, val, 2);
			val += __shfl_down_sync(0xffffffff, val, 1);
		}
		return val;
}

__global__ void tc_sum_int(uint8_t* a, int* out, int N) {
	//int offset = blockIdx.x * TCSQ * 32;       
	int offset = blockIdx.x * BLOCK_SIZE;
	int value = 0;
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += blockDim.x * gridDim.x) {
		value = block_reduce_tc_int_v2(a, N, offset);
		// if (threadIdx.x == 0) {
			
		// 	atomicAdd(out, value);
		// }
	}
}

template<typename T>
__global__ void shfl_sum(T* input, TYPE_NAME *output, int num) {
	//__shared__ T shared[256];

	//for (int i = threadIdx.x; i < 256 / (sizeof(TYPE_NAME) / sizeof(T)); i += blockDim.x) {
	//	((TYPE_NAME*)shared)[i] = ((TYPE_NAME*)input)[i + blockIdx.x * blockDim.x / (sizeof(TYPE_NAME) / sizeof(T))];
	//}
	//__syncthreads();//
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < num; i += blockDim.x * gridDim.x) {
		int val = input[i];// shared[threadIdx.x];// this is fastest
		val = warp_block_sum(val);
		// if (threadIdx.x == 0) {
			
		// 	atomicAdd(output, val);
		// }
	}
}

__global__ void cub_sum(uint8_t* input, int* output, int num) {
	// __shared__ uint8_t shared[256];

	// for (int i = threadIdx.x; i < 64; i += blockDim.x) {
	// 	((int*)shared)[i] = ((int*)input)[i + blockIdx.x * blockDim.x / 4];
	// }
	// __syncthreads();//
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < num; i += blockDim.x * gridDim.x) {
		typedef cub::BlockReduce<int, 256> BlockReduce;
		__shared__ typename BlockReduce::TempStorage temp_storage;
		int thread_data = input[i];//(int)shared[threadIdx.x];
		int val = BlockReduce(temp_storage).Sum(thread_data);

		// if (threadIdx.x == 0) {
		// 	atomicAdd(output, val);
		// }
	}

}

int main(int argc, char **argv) {
	HANDLE_ERROR(cudaSetDevice(3));
	printf("please select which algoritm you choose:\n \
		0. block reduce base on warp shfl\n \
		1. block reduce base on tensor core\n \
		2. block reduce base on cub library(seem same to 0)\n");
	printf("select block num in argv[2]\n");
	
	if(argc != 3){
		printf("error\n");
		return 0;
	}
	int algo, block_num;
	algo = atoi(argv[1]);
	assert(algo < 3);
	block_num = atoi(argv[2]);
	printf("block_num is %d\n", block_num);
	
	const int n = block_num * BLOCK_SIZE;
	TYPE_NAME2* input = new TYPE_NAME2[n];
	TYPE_NAME output;
	initial(input, n);
	TYPE_NAME cpu_out = cpu_sum_int(input, n);

	TYPE_NAME2* d_input;
	TYPE_NAME *d_output;
	half* d_input_half;
	HANDLE_ERROR(cudaMalloc((void**)&d_input, n * sizeof(TYPE_NAME2)));
	HANDLE_ERROR(cudaMalloc((void**)&d_input_half, n * sizeof(half)));
	HANDLE_ERROR(cudaMalloc((void**)&d_output, 1 * sizeof(TYPE_NAME)));
	HANDLE_ERROR(cudaMemcpy(d_input, input, n * sizeof(TYPE_NAME2), cudaMemcpyHostToDevice));


	// for (size_t i = 0; i < 1; i++)
	// {
		if(algo == 0){
			shfl_sum << <block_num, BLOCK_SIZE >> > (d_input, d_output, n);
		}
		else if(algo == 1){
			tc_sum_int << <block_num, BLOCK_SIZE >> > (d_input, d_output, n);
		}
		else{
			cub_sum << <block_num, BLOCK_SIZE >> > (d_input, d_output, n);
		}
		cudaDeviceSynchronize();
		HANDLE_ERROR(cudaGetLastError());

	// }

	std::cout << "cpu output is " << cpu_out << std::endl;
	HANDLE_ERROR(cudaMemcpy(&output, d_output, 1 * sizeof(TYPE_NAME), cudaMemcpyDeviceToHost));
	std::cout << "gpu output is " << output << std::endl;
	return 0;
}