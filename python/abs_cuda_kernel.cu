#include <torch/torch.h>
#include <vector>


__global__ void abs_cuda_forward_kernel(
    const float* input,
    float* output,
    const int n
) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < n) {
    output[index] = fabs(input[index]);
  }
}

std::vector<at::Tensor> abs_cuda_forward(
    torch::Tensor input
) {
  auto output = torch::empty_like(input);

  const int n = input.numel();
  const int blockSize = 256;
  const int numBlocks = (n + blockSize - 1) / blockSize;

  abs_cuda_forward_kernel<<<numBlocks, blockSize>>>(
      input.data_ptr<float>(),
      output.data_ptr<float>(),
      n
  );

  return {output};
}