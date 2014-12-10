#include "cudaCode.cuh"

__device__ int x_error, y_error;
__device__ int errorFlag = 0;
__device__ int errorCode;

__device__ void nanException(int x, int y, int e) {
  errorFlag = 1;
  errorCode = e;
  x_error = x;
  y_error = y;
  asm("trap;");
}

__global__ void calcRhoGPU(float* data, float* rhos) {
  int tid = blockDim.x*blockIdx.x+threadIdx.x + yOffset[0];
  if (tid < gridIter[0]) {
    float result = 0;
    for(int i = 0; i<q; i++) {
      result += data[i*nNodes[0]+tid];
    }
    rhos[tid] = result;
  }
}

__global__ void calcVelGPU(float* data, float* vels, float* usqrs, float* rhos) {

  int tid = blockDim.x*blockIdx.x+threadIdx.x + yOffset[0];

  if (tid < gridIter[0]) {
    int thisnNodes = nNodes[0];
    float ux;
    float uy;
    float uxPlus;
    float uyPlus;
    float uxMinus;
    float uyMinus;
    uxPlus = data[1*thisnNodes+tid] + data[5*thisnNodes+tid] + data[8*thisnNodes+tid];
    uyPlus = data[2*thisnNodes+tid] + data[5*thisnNodes+tid] + data[6*thisnNodes+tid];
    uxMinus = data[3*thisnNodes+tid] + data[6*thisnNodes+tid] + data[7*thisnNodes+tid];
    uyMinus = data[4*thisnNodes+tid] + data[7*thisnNodes+tid] + data[8*thisnNodes+tid];
    ux = (uxPlus - uxMinus)/rhos[tid];
    uy = (uyPlus - uyMinus)/rhos[tid];
    vels[tid] = ux;
    vels[thisnNodes+tid] = uy;
    usqrs[tid] = ux*ux + uy*uy;
  }
}

__global__ void calcEquilibriumGPU(float* equis, float* rhos, float* vels, float* usqrs) {

  int tid = blockDim.x*blockIdx.x+threadIdx.x + yOffset[0];

  if (tid < gridIter[0]) {
    int thisnNodes = nNodes[0];
    float ux = vels[tid];
    float uy = vels[thisnNodes+tid];
    float uSqr = usqrs[tid];
    float rho = rhos[tid];
    for (int i=0; i<q; i++) {
      float velDot = velCol0[i]*ux + velCol1[i]*uy;
      float result = rho * weights[i] * (1.0 + 3.0*velDot + 4.5*velDot*velDot - 1.5*uSqr);
      equis[i*thisnNodes+tid] = result;
    }
  }
}

__global__ void BGKCollideGPU(float* data, float* equis, int* nodeMap) {

  int tid = blockIdx.x*blockDim.x + threadIdx.x + yOffset[0];

  if (tid < gridIter[0]) {
    if (nodeMap[tid] == 0) {
      int thisnNodes = nNodes[0];
      for (int i = 0; i<q; i++) {
        data[i*thisnNodes+tid] *= (1.0 - omega[0]);
        data[i*thisnNodes+tid] += omega[0]*equis[i*thisnNodes+tid];
      }
    }
  }
}

__global__ void bounceBackGPU(float* data, int* nodeMap) {

  int tid = blockIdx.x*blockDim.x + threadIdx.x + yOffset[0];

  if (tid<gridIter[0]) {
    int thisnNodes = nNodes[0];
    float oldVels[q];
    if (nodeMap[tid] == 1) {
      for (int i = 0; i<q; i++) {
        float tmpResult = data[i*thisnNodes+tid];
        // if (tmpResult != tmpResult) {
        //   nanException(tid%lx, tid/lx, 3);
        // }
        oldVels[i] = tmpResult;
      }
      for (int i = 0; i<q; i++) {
        data[i*thisnNodes+tid] = oldVels[bounceBackMap[i]];
      }
    }
  }
}

__global__ void streamGPU(float* outData, float* inData) {

  int tid = blockIdx.x*blockDim.x + threadIdx.x + yOffset[0];

  //Avoid streaming the special first and last column.
  //Probably best avoided by just no-oping those threads.

  if (tid < gridIter[0]) {
    int thisnNodes = nNodes[0];
    int thisLx = lx[0];
    int x = tid % thisLx;
    if (x != 0 || x != thisLx-1) {
      for (int i = 0; i<q; i++) {
        float tmpResult = inData[i*thisnNodes + tid];
        int dst = i*thisnNodes + tid + velColInt1[i]*thisLx + velColInt0[i];
        outData[dst] = tmpResult;
      }
    }
  }
}

__global__ void streamInGPU(float* outData, float* inData, float* inFlow) {

  int tid = blockIdx.x*blockDim.x + threadIdx.x;

  if (tid < gridIter[0]) {
    int thisnNodes = nNodes[0];
    int thisLx = lx[0];
    int thisLy = ly[0];
    int node = tid*thisLx + yOffset[0];
    int y = tid + 1;
    outData[5*thisnNodes+node] = inFlow[(thisLy+2)*5+y];
    outData[1*thisnNodes+node] = inFlow[(thisLy+2)*1+y];
    outData[8*thisnNodes+node] = inFlow[(thisLy+2)*8+y];

    outData[2*thisnNodes+node] = inData[2*thisnNodes+node-thisLx];
    outData[4*thisnNodes+node] = inData[4*thisnNodes+node+thisLx];

    outData[5*thisnNodes+node+1] = inData[5*thisnNodes+node-thisLx];
    outData[1*thisnNodes+node+1] = inData[1*thisnNodes+node];
    outData[8*thisnNodes+node+1] = inData[8*thisnNodes+node+thisLx];
  }
}

__global__ void streamOutGPU(float* outData, float* inData, float* inFlow) {

  int tid = blockIdx.x*blockDim.x + threadIdx.x;

  if (tid < gridIter[0]) {
    int thisnNodes = nNodes[0];
    int thisLx = lx[0];
    int thisLy = ly[0];
    int node = tid*thisLx + yOffset[0] + thisLx-1;
    int y = node / thisLx;
    outData[6*thisnNodes+node] = inFlow[(thisLy+2)*6+y];
    outData[3*thisnNodes+node] = inFlow[(thisLy+2)*3+y];
    outData[7*thisnNodes+node] = inFlow[(thisLy+2)*7+y];

    outData[2*thisnNodes+node] = inData[2*thisnNodes+node-thisLx];
    outData[4*thisnNodes+node] = inData[4*thisnNodes+node+thisLx];

    outData[6*thisnNodes+node-1] = inData[6*thisnNodes+node-thisLx];
    outData[3*thisnNodes+node-1] = inData[3*thisnNodes+node];
    outData[7*thisnNodes+node-1] = inData[7*thisnNodes+node+thisLx];
  }
}