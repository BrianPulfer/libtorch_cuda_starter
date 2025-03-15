#include <torch/extension.h>
#include <vector>
#include <pybind11/pybind11.h>

// CUDA forward declarations
std::vector<torch::Tensor> abs_cuda_forward(torch::Tensor input);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> abs_forward(torch::Tensor input){
  return abs_cuda_forward(input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("abs_forward", &abs_forward, "ABS forward (CUDA)");
  // m.def("backward", &lltm_backward, "LLTM backward (CUDA)");
}