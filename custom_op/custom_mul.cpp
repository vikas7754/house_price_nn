#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/record_function.h>

at::Tensor custom_mul_cpu(const at::Tensor& a, const at::Tensor& b) {
  RECORD_FUNCTION("custom_mul_cpu", std::vector<c10::IValue>({a, b}));
  TORCH_CHECK(a.device().is_cpu() && b.device().is_cpu(), "custom_mul: tensors must be on CPU");
  return at::mul(a, b);
}

at::Tensor custom_mul_cuda(const at::Tensor& a, const at::Tensor& b) {
  RECORD_FUNCTION("custom_mul_cuda", std::vector<c10::IValue>({a, b}));
  TORCH_CHECK(a.device().is_cuda() && b.device().is_cuda(), "custom_mul: tensors must be on CUDA");
  return at::mul(a, b);
}

TORCH_LIBRARY(my_mul_ops, m) {
  m.def("custom_mul(Tensor a, Tensor b) -> Tensor");
}

TORCH_LIBRARY_IMPL(my_mul_ops, CPU, m) {
  m.impl("custom_mul", &custom_mul_cpu);
}

TORCH_LIBRARY_IMPL(my_mul_ops, CUDA, m) {
  m.impl("custom_mul", &custom_mul_cuda);
}

TORCH_LIBRARY_IMPL(my_mul_ops, Autograd, m) {
  m.impl("custom_mul", torch::CppFunction::makeFallthrough());
}
