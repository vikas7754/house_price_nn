#include <torch/script.h>
#include <torch/autograd/profiler.h>

torch::Tensor custom_add(torch::Tensor a, torch::Tensor b) {
  RECORD_FUNCTION("custom_add_op", std::vector<c10::IValue>({a, b}));
  return a + b;
}

static auto registry =
  torch::RegisterOperators("my_ops::custom_add", &custom_add);
