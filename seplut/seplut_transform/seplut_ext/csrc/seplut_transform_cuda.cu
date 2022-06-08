#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>


#define CUDA_1D_KERNEL_LOOP(i, n)                                \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
         i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 512

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int max_block_num = at::cuda::getCurrentDeviceProperties()->maxGridSize[0];
  return min(optimal_block_num, max_block_num);
}


/* std::clamp is only available since c++17 */
template <typename scalar_t>
inline __device__ constexpr const scalar_t& clamp(
    const scalar_t& v, const scalar_t& lo, const scalar_t& hi)
{
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}


template <typename scalar_t>
__launch_bounds__(THREADS_PER_BLOCK)
__global__ void seplut_transform_cuda_forward_kernel(
        const int n,
        const scalar_t* __restrict__ data_inp,
        const scalar_t* __restrict__ data_lut3d,
        const scalar_t* __restrict__ data_lut1d,
        const int height,
        const int width,
        const int num_channels,
        const int num_3d_vertices,
        const int num_1d_vertices,
        scalar_t* __restrict__ data_col) {

    const scalar_t size_1d_bin = 1.0 / (num_1d_vertices - 1);
    const scalar_t size_3d_bin = 1.0 / (num_3d_vertices - 1);

    CUDA_1D_KERNEL_LOOP(index, n) {

        /* retrieve rgb value of the pixel */
        const scalar_t r = data_inp[index];
        const scalar_t g = data_inp[index + height * width];
        const scalar_t b = data_inp[index + height * width * 2];

        /* retrieve index of the interpolation vertices */
        const int32_t rid_1d = clamp((int32_t)floor(r * (num_1d_vertices - 1)), 0, num_1d_vertices - 2);
        const int32_t gid_1d = clamp((int32_t)floor(g * (num_1d_vertices - 1)), 0, num_1d_vertices - 2);
        const int32_t bid_1d = clamp((int32_t)floor(b * (num_1d_vertices - 1)), 0, num_1d_vertices - 2);

        /* retrieve the interpolation vertices (number of 8 in case of trilinear interpolation) */
        const int rid_1d_0 = rid_1d;
        const int rid_1d_1 = rid_1d + 1;
        const int gid_1d_0 = gid_1d + num_1d_vertices;
        const int gid_1d_1 = gid_1d + 1 + num_1d_vertices;
        const int bid_1d_0 = bid_1d + num_1d_vertices * 2;
        const int bid_1d_1 = bid_1d + 1 + num_1d_vertices * 2;

        /* compute interpolation weights */
        const scalar_t rd_1d = (r - size_1d_bin * rid_1d) / size_1d_bin;
        const scalar_t gd_1d = (g - size_1d_bin * gid_1d) / size_1d_bin;
        const scalar_t bd_1d = (b - size_1d_bin * bid_1d) / size_1d_bin;

        const scalar_t rw0_1d = (1 - rd_1d);
        const scalar_t rw1_1d = (    rd_1d);
        const scalar_t gw0_1d = (1 - gd_1d);
        const scalar_t gw1_1d = (    gd_1d);
        const scalar_t bw0_1d = (1 - bd_1d);
        const scalar_t bw1_1d = (    bd_1d);

        const scalar_t r_inter = rw0_1d * data_lut1d[rid_1d_0] + rw1_1d * data_lut1d[rid_1d_1];
        const scalar_t g_inter = gw0_1d * data_lut1d[gid_1d_0] + gw1_1d * data_lut1d[gid_1d_1];
        const scalar_t b_inter = bw0_1d * data_lut1d[bid_1d_0] + bw1_1d * data_lut1d[bid_1d_1];

        /* retrieve index of the interpolation vertices */
        const int32_t rid_3d = clamp((int32_t)floor(r_inter * (num_3d_vertices - 1)), 0, num_3d_vertices - 2);
        const int32_t gid_3d = clamp((int32_t)floor(g_inter * (num_3d_vertices - 1)), 0, num_3d_vertices - 2);
        const int32_t bid_3d = clamp((int32_t)floor(b_inter * (num_3d_vertices - 1)), 0, num_3d_vertices - 2);

        /* utility variables for indexing */
        const int num_3d_vertices_2 = num_3d_vertices * num_3d_vertices;
        const int num_3d_vertices_3 = num_3d_vertices_2 * num_3d_vertices;

        /* retrieve the interpolation vertices (number of 8 in case of trilinear interpolation) */
        const int id000 = (rid_3d    ) + num_3d_vertices * (gid_3d    ) + num_3d_vertices_2 * (bid_3d    );
        const int id100 = (rid_3d + 1) + num_3d_vertices * (gid_3d    ) + num_3d_vertices_2 * (bid_3d    );
        const int id010 = (rid_3d    ) + num_3d_vertices * (gid_3d + 1) + num_3d_vertices_2 * (bid_3d    );
        const int id110 = (rid_3d + 1) + num_3d_vertices * (gid_3d + 1) + num_3d_vertices_2 * (bid_3d    );
        const int id001 = (rid_3d    ) + num_3d_vertices * (gid_3d    ) + num_3d_vertices_2 * (bid_3d + 1);
        const int id101 = (rid_3d + 1) + num_3d_vertices * (gid_3d    ) + num_3d_vertices_2 * (bid_3d + 1);
        const int id011 = (rid_3d    ) + num_3d_vertices * (gid_3d + 1) + num_3d_vertices_2 * (bid_3d + 1);
        const int id111 = (rid_3d + 1) + num_3d_vertices * (gid_3d + 1) + num_3d_vertices_2 * (bid_3d + 1);

        /* compute interpolation weights */
        const scalar_t rd_3d = (r_inter - size_3d_bin * rid_3d) / size_3d_bin;
        const scalar_t gd_3d = (g_inter - size_3d_bin * gid_3d) / size_3d_bin;
        const scalar_t bd_3d = (b_inter - size_3d_bin * bid_3d) / size_3d_bin;

        const scalar_t w000 = (1 - rd_3d) * (1 - gd_3d) * (1 - bd_3d);
        const scalar_t w100 = (    rd_3d) * (1 - gd_3d) * (1 - bd_3d);
        const scalar_t w010 = (1 - rd_3d) * (    gd_3d) * (1 - bd_3d);
        const scalar_t w110 = (    rd_3d) * (    gd_3d) * (1 - bd_3d);
        const scalar_t w001 = (1 - rd_3d) * (1 - gd_3d) * (    bd_3d);
        const scalar_t w101 = (    rd_3d) * (1 - gd_3d) * (    bd_3d);
        const scalar_t w011 = (1 - rd_3d) * (    gd_3d) * (    bd_3d);
        const scalar_t w111 = (    rd_3d) * (    gd_3d) * (    bd_3d);

        /* Execute the interpolation */
        for (int i = 0; i < num_channels; ++i) {
            data_col[index + height * width * i] = 
                w000 * data_lut3d[id000 + num_3d_vertices_3 * i] + w100 * data_lut3d[id100 + num_3d_vertices_3 * i] + 
                w010 * data_lut3d[id010 + num_3d_vertices_3 * i] + w110 * data_lut3d[id110 + num_3d_vertices_3 * i] + 
                w001 * data_lut3d[id001 + num_3d_vertices_3 * i] + w101 * data_lut3d[id101 + num_3d_vertices_3 * i] + 
                w011 * data_lut3d[id011 + num_3d_vertices_3 * i] + w111 * data_lut3d[id111 + num_3d_vertices_3 * i];
        }
    }
}


void SepLUTTransformForwardCUDAKernelLauncher(
    const torch::Tensor &input, const torch::Tensor &lut3d,
    const torch::Tensor &lut1d, torch::Tensor output) {

    /* tensor check
       input: (b,3,h,w); lut3d: (b,m,d,d,d), lut1d: (b,3,d), output: (b,m,h,w)
     */

    c10::cuda::CUDAGuard device_guard(input.device());

    /* retrieve some meta-information of the input tensors */
    int batch_size = input.size(0);
    int height     = input.size(2);
    int width      = input.size(3);

    int num_channels = lut3d.size(1);
    int stride_lut3d = lut3d.size(2);
    int stride_lut1d = lut1d.size(2);

    int num_kernels = height * width;
    for (int elt = 0; elt < batch_size; ++elt) {

        /* launch the CUDA kernel */
        AT_DISPATCH_FLOATING_TYPES(
            input.scalar_type(), "seplut_transform_cuda_forward", ([&] {
                const scalar_t *data_inp = input[elt].data_ptr<scalar_t>();
                const scalar_t *data_lut3d = lut3d[elt].data_ptr<scalar_t>();
                const scalar_t *data_lut1d = lut1d[elt].data_ptr<scalar_t>();
                scalar_t *data_col = output[elt].data_ptr<scalar_t>();

                seplut_transform_cuda_forward_kernel<<<GET_BLOCKS(num_kernels),
                                                    THREADS_PER_BLOCK, 0,
                                                    at::cuda::getCurrentCUDAStream()>>>(
                    num_kernels, data_inp, data_lut3d, data_lut1d,
                    height, width, num_channels, stride_lut3d, stride_lut1d,
                    data_col);
            }));

        AT_CUDA_CHECK(cudaGetLastError());
    }
}
