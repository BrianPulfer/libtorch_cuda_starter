#include <cuda.h>
#include <cuda_runtime.h>

#include <torch/torch.h>
#include <iostream>

using namespace std;

__global__ void sumKernel(const float* src, float* dst, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n){ 
        atomicAdd(dst, src[idx]);
    }
}

float sumInterface(const torch::Tensor& src){
    int n = src.numel();
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Creating float (0.0) on device
    float *result = new float(0.0f);
    float *dst;
    cudaMalloc(&dst, sizeof(float));
    cudaMemcpy(dst, result, sizeof(float), cudaMemcpyHostToDevice);

    // Launching kernel
    sumKernel<<<numBlocks, blockSize>>>(src.data_ptr<float>(), dst, n);
    
    // Copying back to host
    cudaMemcpy(result, dst, sizeof(float), cudaMemcpyDeviceToHost);
    return *result;
}

int main(int argc, char* argv[]){
    torch::Tensor tensor = torch::randn({2, 3}).to(at::kCUDA);

    cout << "Kernel Sum: " << sumInterface(tensor) << endl;
    cout << "Torchlib Sum: " << tensor.sum().item<float>() << endl;
    return 0;
}