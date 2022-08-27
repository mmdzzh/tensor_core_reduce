
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
	//if (warpIdx == 0) return warp_sum_utils(val);
}

//template<typename T>
__inline__ __device__ float reduction_tc_warp(int N, half* A, int offset, int lane, int warpoff) {
	// definicion de offset y fragmentos
	wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
	wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> d_frag;
	wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> r_frag;

	// (1) cargar datos de memoria global a A, B y C frags
	wmma::fill_fragment(a_frag, 1.0f);
	//wmma::fill_fragment(b_frag, 0.0f);
	wmma::fill_fragment(d_frag, 0.0f);

	// (2) mejora MMA multiples encadenados
	//const int bigoffset = gridDim.x * 32 * TCSQ;
	//if(offset >= N){ return 0.0f; }
//#pragma loop unroll
//	for (int i = 0; i < R; ++i) {
		
		wmma::load_matrix_sync(b_frag, A + offset, TCSIZE);
		wmma::mma_sync(d_frag, a_frag, b_frag, d_frag);
	//}

	// (3) preparando datos para segundo MMA
	wmma::fill_fragment(b_frag, 1.0f);
	// [OPCION 1] copia manual
	//#pragma loop unroll
	//for(int i=0; i < 8 ; ++i){
	//    a_frag.x[i] = d_frag.x[i];
	//    a_frag.x[i+8] = d_frag.x[i];
	//}

	//int offwid = (threadIdx.x/32)*256;
	// [OPCION 2] copia a shared mem

	__shared__ half As[16];
	wmma::store_matrix_sync(As, d_frag, TCSIZE, wmma::mem_row_major);
	wmma::load_matrix_sync(a_frag, As, TCSIZE);
	wmma::fill_fragment(d_frag, 0.0f);


	

	//// (4) MMA
	wmma::mma_sync(r_frag, a_frag, b_frag, d_frag);
	//return __float2float(d_frag.x[lane]);
	// (5) Almacenar resultado
	if (lane == 0) {
		return r_frag.x[0];
	}
	//	//printf("block: %i, val %f\n",blockIdx.x,(float)d_frag.x[0]);
	//	//printf("%f\n",(float)d_frag.x[0]);
	//	return __float2float(d_frag.x[0]);
	//	//return 1.0f;
	//}
	else return 0.0f;
}

//template<typename T>
__inline__ __device__ float block_reduce_tc(int N, half* a, int offset) {
	__shared__ float shared[WARPSIZE];
	int tid = threadIdx.x;
	int lane = tid & (WARPSIZE - 1);
	//int wid = tid/WARPSIZE;
	int wid = tid >> 5;
	float val =  reduction_tc_warp(N, a, offset + wid * WARPSIZE, lane, 0);
	
	//printf("%i, %i, %i\n", threadIdx.x, wid, lane);
	//return val;
	//if (lane == 0 || lane == 16) {
	//	shared[wid + lane/2] = val;
	//	printf("wid %i val %f\n", wid+lane/2, val);
	//}
	//__syncthreads();
	//val = (tid < blockDim.x/WARPSIZE) ? shared[lane] : (T) 0.0f;
	
	/*val = (tid < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
	if (wid == 0) {
		return  warp_sum_utils(val);
	}*/
	if (threadIdx.x == 0) return val;
}

__inline__ __device__ int reduction_tc_warp_int(int N, uint8_t* a, int offset, int lane, int wid) {
	// definicion de offset y fragmentos
	wmma::fragment<wmma::matrix_a, WMMA_M_INT, WMMA_N_INT, WMMA_K_INT, uint8_t, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, WMMA_M_INT, WMMA_N_INT, WMMA_K_INT, uint8_t, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, WMMA_M_INT, WMMA_N_INT, WMMA_K_INT, int> d_frag;
	//wmma::fragment<wmma::accumulator, WMMA_M_INT, WMMA_N_INT, WMMA_K_INT, int> r_frag;

	// (1) cargar datos de memoria global a A, B y C frags
	wmma::fill_fragment(a_frag, 1);
	//wmma::fill_fragment(b_frag, 0.0f);
	wmma::fill_fragment(d_frag, 0);

	// (2) mejora MMA multiples encadenados
	//const int bigoffset = gridDim.x * 32 * TCSQ;
	//if(offset >= N){ return 0.0f; }
//#pragma loop unroll
//	for (int i = 0; i < R; ++i) {

	wmma::load_matrix_sync(b_frag, a + offset, TCSIZE_INT8);
	wmma::mma_sync(d_frag, a_frag, b_frag, d_frag);
	//printf("wid is : %i, %i, b_frag is : %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d\n a_frag is %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d\n", wid, lane,
	//	b_frag.x[0],  b_frag.x[1],  b_frag.x[2],  b_frag.x[3],  b_frag.x[16], b_frag.x[17], b_frag.x[18], b_frag.x[19], b_frag.x[8],
	//	b_frag.x[9], b_frag.x[10], b_frag.x[11], b_frag.x[12], b_frag.x[13], b_frag.x[14], b_frag.x[15],
	//	a_frag.x[0], a_frag.x[1], a_frag.x[2], a_frag.x[3], a_frag.x[4], a_frag.x[5], a_frag.x[6], a_frag.x[7], a_frag.x[16],
	//	a_frag.x[17], a_frag.x[18], a_frag.x[19], a_frag.x[12], a_frag.x[13], a_frag.x[14], a_frag.x[15]);
		//printf("wid is : %i, %i, b_frag is : %d, %d\n", wid, lane, b_frag.x[0], a_frag.x[0]);
	//}
	//printf("wid: %i, %i, %i, val %d, %d, %d, %d %d, %d, %d, %d\n", wid, lane, threadIdx.x, d_frag.x[0], d_frag.x[1], d_frag.x[2], d_frag.x[3], d_frag.x[4], d_frag.x[5], d_frag.x[6], d_frag.x[7]);
	// (3) preparando datos para segundo MMA

	// [OPCION 1] copia manual
	//#pragma loop unroll
	//for(int i=0; i < 8 ; ++i){
	//    a_frag.x[i] = d_frag.x[i];
	//    a_frag.x[i+8] = d_frag.x[i];
	//}

	//int offwid = (threadIdx.x/32)*256;
	// [OPCION 2] copia a shared mem
	//printf("warp offset %i \n", warpoff);
	// 
	// 	wmma::fill_fragment(b_frag, 1);
	//__shared__ uint8_t As[16];
	//wmma::store_matrix_sync(As, d_frag, 16, wmma::mem_row_major);
	//wmma::load_matrix_sync(a_frag, As, 16);
	//wmma::fill_fragment(d_frag, 0);
	//

	

	//// (4) MMA
	//wmma::mma_sync(r_frag, a_frag, b_frag, r_frag);

	// (5) Almacenar resultado
	//if (lane == 0) {
	////	//printf("block: %i, val %f\n",blockIdx.x,(float)d_frag.x[0]);
	////	//printf("%f\n",(float)d_frag.x[0]);
	////	int val = 
	//	return r_frag.x[0];
	////	//return 1.0f;
	//}
	//else return 0;

	//int val = threadIdx.x < 16 ? 

	///success
	__shared__ int shared[16];

	int val = d_frag.x[0] + d_frag.x[1];
	
	if (lane == 0) {
		printf("block: %i, %i, %i, val %d, %d\n", wid, lane,threadIdx.x, b_frag.x[0], b_frag.x[1]);
		shared[wid] = val;
	}
	__syncthreads();
	val = threadIdx.x < 8 ? shared[threadIdx.x] : 0;
	
	//printf("block: %i, %i, val %d, %d\n", blockIdx.x, threadIdx.x, val, shared[threadIdx.x]);
	if(wid == 0) return  warp_sum_utils(val);
	///////////////////
}

template<typename T>
__inline__ __device__ int block_reduce_tc_int(T* a, int N, int offset) {
	__shared__ int shared[WARPSIZE];
	int tid = threadIdx.x;
	int lane = tid & (WARPSIZE - 1);
	//int wid = tid/WARPSIZE;
	int wid = tid >> 5;
	int val = reduction_tc_warp_int(N, a, offset + wid * WARPSIZE, lane, wid);

	//return val;
	//if (lane == 0) {
	//	shared[wid] = val;
	//}
	//__syncthreads();
	//val = (tid < blockDim.x/WARPSIZE) ? shared[lane] : (T) 0.0f;
	//printf("thread %i val %f\n", threadIdx.x, val);
	//val = (tid < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
	//if (wid == 0) {
	//	val = warp_sum_utils(val);
	//}
	if (threadIdx.x == 0) return val;
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
			//printf("%d, %d, %d, %d, %d, %d, %d, %d, %d\n", b_frag.x[0], b_frag.x[1], b_frag.x[2], b_frag.x[3], b_frag.x[4], b_frag.x[5], b_frag.x[6], b_frag.x[7]);
			shared[wid] = val;
		}
		__syncthreads();
		 val = threadIdx.x < 8 ? shared[threadIdx.x] + shared[threadIdx.x + 8] : 0;
		if (wid == 0) {
			//val += __shfl_xor_sync(0xffffffff, val, 16);
			//val += __shfl_xor_sync(0xffffffff, val, 8);// warp_sum_utils(val);
			val += __shfl_down_sync(0xffffffff, val, 4);
			//val += __shfl_down_sync(0xffffffff, val, 2);
			//val += __shfl_down_sync(0xffffffff, val, 1);
		}
		return val;
}

/// </summary>
/// <typeparam name="T"></typeparam>
/// <param name="val"></param>
/// <returns></returns>
//template<typename T>
__global__ void tc_sum(half* a, float* out, int N) {
	//int offset = blockIdx.x * TCSQ * 32;       
	int offset = blockIdx.x *BLOCK_SIZE;
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += blockDim.x * gridDim.x) {
		float sumf = block_reduce_tc(N, a, offset);
		//if((threadIdx.x & 31) == 0){
		//    //printf("offset %i \n",offset);
		//    atomicAdd(out, sumf);
		//}
		if (threadIdx.x == 0) {
			atomicAdd(out, sumf);
		}
	}
}

__global__ void tc_sum_int(uint8_t* a, int* out, int N) {
	//int offset = blockIdx.x * TCSQ * 32;       
	int offset = blockIdx.x * BLOCK_SIZE;
	int sumf = 0;
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N; i += blockDim.x * gridDim.x) {
		sumf = block_reduce_tc_int_v2(a, N, offset);

		
		//if (threadIdx.x == 0) {
		//	
		//	atomicAdd(out, sumf);
		//}
	}
}

template<typename T>
__global__ void shfl_sum(T* input, TYPE_NAME *output, int num) {
	//__shared__ T shared[256];

	//for (int i = threadIdx.x; i < 256 / (sizeof(TYPE_NAME) / sizeof(T)); i += blockDim.x) {
	//	((TYPE_NAME*)shared)[i] = ((TYPE_NAME*)input)[i + blockIdx.x * blockDim.x / (sizeof(TYPE_NAME) / sizeof(T))];
	//}
	//__syncthreads();
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < num; i += blockDim.x * gridDim.x) {
		int val = input[i];// shared[threadIdx.x];
		val = warp_block_sum(val);
		//if (threadIdx.x == 0) {
		//	
		//	atomicAdd(output, val);
		//}
	}
}

__global__ void cub_sum(uint8_t* input, int* output, int num) {
	for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < num; i += blockDim.x * gridDim.x) {
		typedef cub::BlockReduce<int, 256> BlockReduce;
		__shared__ typename BlockReduce::TempStorage temp_storage;
		int thread_data = input[i];
		int val = BlockReduce(temp_storage).Sum(thread_data);

		if (threadIdx.x == 0) {
			atomicAdd(output, val);
		}
	}

}

int main() {
	TYPE_NAME2* input = new TYPE_NAME2[BLOCK_NUM * BLOCK_SIZE];
	TYPE_NAME output;
	initial(input, BLOCK_NUM * BLOCK_SIZE);
	TYPE_NAME cpu_out = cpu_sum_int(input, BLOCK_NUM * BLOCK_SIZE);

	TYPE_NAME2* d_input;
	TYPE_NAME *d_output;
	half* d_input_half;
	HANDLE_ERROR(cudaMalloc((void**)&d_input, BLOCK_NUM * BLOCK_SIZE * sizeof(TYPE_NAME2)));
	HANDLE_ERROR(cudaMalloc((void**)&d_input_half, BLOCK_NUM * BLOCK_SIZE * sizeof(half)));
	HANDLE_ERROR(cudaMalloc((void**)&d_output, 1 * sizeof(TYPE_NAME)));
	HANDLE_ERROR(cudaMemcpy(d_input, input, BLOCK_NUM * BLOCK_SIZE * sizeof(TYPE_NAME2), cudaMemcpyHostToDevice));

	/*convertFp32ToFp16 << <BLOCK_NUM, BLOCK_SIZE >> > (d_input_half, d_input, BLOCK_NUM * BLOCK_SIZE);
	cudaDeviceSynchronize();
	HANDLE_ERROR(cudaGetLastError());*/
	for (size_t i = 0; i < 1; i++)
	{
		//shfl_sum << <BLOCK_NUM, BLOCK_SIZE >> > (d_input, d_output, BLOCK_NUM * BLOCK_SIZE);
		//cudaDeviceSynchronize();
		//HANDLE_ERROR(cudaGetLastError());
	 

		//tc_sum << <BLOCK_NUM, BLOCK_SIZE >> > (d_input, d_output, BLOCK_NUM * BLOCK_SIZE);
		//cudaDeviceSynchronize();
		//HANDLE_ERROR(cudaGetLastError());

		tc_sum_int << <BLOCK_NUM, BLOCK_SIZE >> > (d_input, d_output, BLOCK_NUM * BLOCK_SIZE);
		cudaDeviceSynchronize();
		HANDLE_ERROR(cudaGetLastError());

		//cub_sum << <BLOCK_NUM, BLOCK_SIZE >> > (d_input, d_output, BLOCK_NUM * BLOCK_SIZE);
		//cudaDeviceSynchronize();
		//HANDLE_ERROR(cudaGetLastError());
	}

	std::cout << "cpu output is " << cpu_out << std::endl;
	HANDLE_ERROR(cudaMemcpy(&output, d_output, 1 * sizeof(TYPE_NAME), cudaMemcpyDeviceToHost));
	std::cout << "gpu output is " << output << std::endl;
	return 0;
}