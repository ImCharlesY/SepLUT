#include <torch/extension.h>

/* CUDA Forward Declarations */

void SepLUTTransformForwardCUDAKernelLauncher(
    const torch::Tensor &input, const torch::Tensor &lut3d,
    const torch::Tensor &lut1d, torch::Tensor output);


void seplut_transform_cuda_forward(
    const torch::Tensor &input,
    const torch::Tensor &lut3d,
    const torch::Tensor &lut1d,
    torch::Tensor output) {

    SepLUTTransformForwardCUDAKernelLauncher(input, lut3d, lut1d, output);
}


void seplut_transform_cpu_forward(
    const torch::Tensor &input,
    const torch::Tensor &lut3d,
    const torch::Tensor &lut1d,
    torch::Tensor output);


/* C++ Interfaces */

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void seplut_transform_forward(
    const torch::Tensor &input,
    const torch::Tensor &lut3d,
    const torch::Tensor &lut1d,
    torch::Tensor output) {
    
    if (input.device().is_cuda()) {
        CHECK_INPUT(input);
        CHECK_INPUT(lut3d);
        CHECK_INPUT(lut1d);
        CHECK_INPUT(output);

        seplut_transform_cuda_forward(input, lut3d, lut1d, output);
    } else {
        CHECK_CONTIGUOUS(input);
        CHECK_CONTIGUOUS(lut3d);
        CHECK_CONTIGUOUS(lut1d);
        CHECK_CONTIGUOUS(output);

        seplut_transform_cpu_forward(input, lut3d, lut1d, output);
    }
}


/* Interfaces Binding */

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cforward", &seplut_transform_forward, "SepLUT-Transform forward");
}

