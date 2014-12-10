#ifndef CUDA_LBM_INCLUDES
#define CUDA_LBM_INCLUDES

// Put the static information about the simulation in GPU constant memory.
__device__ __constant__ float velCol0[9];
__device__ __constant__ float velCol1[9];
__device__ __constant__ int velColInt0[9];
__device__ __constant__ int velColInt1[9];
__device__ __constant__ float weights[9];
__device__ __constant__ int bounceBackMap[9];
__device__ __constant__ float omega[1];
__device__ __constant__ int lx[1];
__device__ __constant__ int ly[1];
__device__ __constant__ int yOffset[1];
__device__ __constant__ int gridIter[1];
__device__ __constant__ int nNodes[1];


// We let q be defined at compile time.
const int q = 9;

/*------------------------------------------------------------*/

// Number crunching. The actual work.

// All functions should be called with total number of threads
// (blocks*threadsPerBlock) equalling the total number of nodes (nNodes)
// EXCEPT WHERE OTHERWISE STATED.

__global__ void calcRhoGPU(float* data, float* rhos);

__global__ void calcVelGPU(float* data, float* vels, float* usqrs, float* rhos);

__global__ void calcEquilibriumGPU(float* equis, float* rhos, float* vels, float* usqrs);

__global__ void BGKCollideGPU(float* data, float* equis, int* nodeMap);

__global__ void bounceBackGPU(float* data, int* nodeMap);

__global__ void streamGPU(float* outData, float* inData);

// NOTE: These functions should be called with total number of threads
// (blocks*threadsPerBlock) equalling number of nodes in a column.

__global__ void streamInGPU(float* outData, float* inData, float* inFlow);

__global__ void streamOutGPU(float* outData, float* inData, float* inFlow);

#endif