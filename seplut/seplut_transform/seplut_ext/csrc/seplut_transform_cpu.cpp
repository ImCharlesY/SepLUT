#include <torch/extension.h>

#include <ATen/ATen.h>


/* std::clamp is only available since c++17 */
template <typename scalar_t>
inline constexpr const scalar_t& clamp(
    const scalar_t& v, const scalar_t& lo, const scalar_t& hi)
{
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}


template <typename scalar_t>
void seplut_transform_cpu_forward_impl(
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

    for (int index = 0; index < n; ++index) {

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


template <uint8_t n_vertices, uint8_t bits_shift>
void seplut_transform_cpu_forward_uint8_impl(
    const int n,
    const uint8_t* __restrict__ data_inp,
    const uint8_t* __restrict__ data_lut3d,
    const uint8_t* __restrict__ data_lut1d,
    const int height,
    const int width,
    const int num_channels,
    uint8_t* __restrict__ data_col) {

    constexpr uint32_t norm_term = 255 * 255 * 255;

    for (int index = 0; index < n; ++index) {

        /* retrieve rgb value of the pixel */
        const uint8_t r = data_inp[index];
        const uint8_t g = data_inp[index + height * width];
        const uint8_t b = data_inp[index + height * width * 2];

        /* retrieve index of the interpolation vertices */
        const uint8_t rid_1d = r >> bits_shift;
        const uint8_t gid_1d = g >> bits_shift;
        const uint8_t bid_1d = b >> bits_shift;

        /* retrieve the interpolation vertices (number of 8 in case of trilinear interpolation) */
        const int rid_1d_0 = rid_1d;
        const int rid_1d_1 = rid_1d + 1;
        const int gid_1d_0 = gid_1d + n_vertices;
        const int gid_1d_1 = gid_1d + 1 + n_vertices;
        const int bid_1d_0 = bid_1d + n_vertices * 2;
        const int bid_1d_1 = bid_1d + 1 + n_vertices * 2;

        /* compute interpolation weights */
        const uint8_t rd_1d = ((r << (8 - bits_shift)) - rid_1d * 255);
        const uint8_t gd_1d = ((g << (8 - bits_shift)) - gid_1d * 255);
        const uint8_t bd_1d = ((b << (8 - bits_shift)) - bid_1d * 255);

        const uint8_t rw0_1d = (255 - rd_1d);
        const uint8_t rw1_1d = (      rd_1d);
        const uint8_t gw0_1d = (255 - gd_1d);
        const uint8_t gw1_1d = (      gd_1d);
        const uint8_t bw0_1d = (255 - bd_1d);
        const uint8_t bw1_1d = (      bd_1d);

        const uint8_t r_inter = (rw0_1d * data_lut1d[rid_1d_0] + rw1_1d * data_lut1d[rid_1d_1]) / 255;
        const uint8_t g_inter = (gw0_1d * data_lut1d[gid_1d_0] + gw1_1d * data_lut1d[gid_1d_1]) / 255;
        const uint8_t b_inter = (bw0_1d * data_lut1d[bid_1d_0] + bw1_1d * data_lut1d[bid_1d_1]) / 255;

        /* retrieve index of the interpolation vertices */
        const uint8_t rid_3d = r_inter >> bits_shift;
        const uint8_t gid_3d = g_inter >> bits_shift;
        const uint8_t bid_3d = b_inter >> bits_shift;

        /* utility variables for indexing */
        const int n_vertices_2 = n_vertices * n_vertices;
        const int n_vertices_3 = n_vertices_2 * n_vertices;

        /* retrieve the interpolation vertices (number of 8 in case of trilinear interpolation) */
        const int id000 = (rid_3d    ) + n_vertices * (gid_3d    ) + n_vertices_2 * (bid_3d    );
        const int id100 = (rid_3d + 1) + n_vertices * (gid_3d    ) + n_vertices_2 * (bid_3d    );
        const int id010 = (rid_3d    ) + n_vertices * (gid_3d + 1) + n_vertices_2 * (bid_3d    );
        const int id110 = (rid_3d + 1) + n_vertices * (gid_3d + 1) + n_vertices_2 * (bid_3d    );
        const int id001 = (rid_3d    ) + n_vertices * (gid_3d    ) + n_vertices_2 * (bid_3d + 1);
        const int id101 = (rid_3d + 1) + n_vertices * (gid_3d    ) + n_vertices_2 * (bid_3d + 1);
        const int id011 = (rid_3d    ) + n_vertices * (gid_3d + 1) + n_vertices_2 * (bid_3d + 1);
        const int id111 = (rid_3d + 1) + n_vertices * (gid_3d + 1) + n_vertices_2 * (bid_3d + 1);

        /* compute interpolation weights */
        const uint8_t rd_3d = ((r_inter << (8 - bits_shift)) - rid_3d * 255);
        const uint8_t gd_3d = ((g_inter << (8 - bits_shift)) - gid_3d * 255);
        const uint8_t bd_3d = ((b_inter << (8 - bits_shift)) - bid_3d * 255);

        const uint32_t w000 = (255 - rd_3d) * (255 - gd_3d) * (255 - bd_3d);
        const uint32_t w100 = (      rd_3d) * (255 - gd_3d) * (255 - bd_3d);
        const uint32_t w010 = (255 - rd_3d) * (      gd_3d) * (255 - bd_3d);
        const uint32_t w110 = (      rd_3d) * (      gd_3d) * (255 - bd_3d);
        const uint32_t w001 = (255 - rd_3d) * (255 - gd_3d) * (      bd_3d);
        const uint32_t w101 = (      rd_3d) * (255 - gd_3d) * (      bd_3d);
        const uint32_t w011 = (255 - rd_3d) * (      gd_3d) * (      bd_3d);
        const uint32_t w111 = (      rd_3d) * (      gd_3d) * (      bd_3d);

        /* Execute the interpolation */
        for (int i = 0; i < num_channels; ++i) {
            data_col[index + height * width * i] =
               (w000 * data_lut3d[id000 + n_vertices_3 * i] + w100 * data_lut3d[id100 + n_vertices_3 * i] +
                w010 * data_lut3d[id010 + n_vertices_3 * i] + w110 * data_lut3d[id110 + n_vertices_3 * i] +
                w001 * data_lut3d[id001 + n_vertices_3 * i] + w101 * data_lut3d[id101 + n_vertices_3 * i] +
                w011 * data_lut3d[id011 + n_vertices_3 * i] + w111 * data_lut3d[id111 + n_vertices_3 * i]) / norm_term;
        }
    }
}


void seplut_transform_cpu_forward(
    const torch::Tensor &input,
    const torch::Tensor &lut3d,
    const torch::Tensor &lut1d,
    torch::Tensor output) {

    /* retrieve some meta-information of the input tensors */
    int batch_size = input.size(0);
    int height     = input.size(2);
    int width      = input.size(3);

    int num_channels = lut3d.size(1);
    int stride_lut3d = lut3d.size(2);
    int stride_lut1d = lut1d.size(2);

    int num_kernels = height * width;

    for (int elt = 0; elt < batch_size; ++elt) {
        if (input.scalar_type() == at::ScalarType::Byte
            && stride_lut3d == 17 && stride_lut1d == 17) {
            const uint8_t *data_inp = input[elt].data_ptr<uint8_t>();
            const uint8_t *data_lut3d = lut3d[elt].data_ptr<uint8_t>();
            const uint8_t *data_lut1d = lut1d[elt].data_ptr<uint8_t>();
            uint8_t *data_col = output[elt].data_ptr<uint8_t>();

            seplut_transform_cpu_forward_uint8_impl
                </*n_vertices=*/17, /*bits_shift=*/4>(
                num_kernels, data_inp, data_lut3d, data_lut1d,
                height, width, num_channels, data_col);
        }
        else if (input.scalar_type() == at::ScalarType::Byte
            && stride_lut3d == 9 && stride_lut1d == 9) {
            const uint8_t *data_inp = input[elt].data_ptr<uint8_t>();
            const uint8_t *data_lut3d = lut3d[elt].data_ptr<uint8_t>();
            const uint8_t *data_lut1d = lut1d[elt].data_ptr<uint8_t>();
            uint8_t *data_col = output[elt].data_ptr<uint8_t>();

            seplut_transform_cpu_forward_uint8_impl
                </*n_vertices=*/9, /*bits_shift=*/5>(
                num_kernels, data_inp, data_lut3d, data_lut1d,
                height, width, num_channels, data_col);
        }
        else {
            AT_DISPATCH_FLOATING_TYPES(
                input.scalar_type(), "seplut_transform_cpu_forward", ([&] {
                    const scalar_t *data_inp = input[elt].data_ptr<scalar_t>();
                    const scalar_t *data_lut3d = lut3d[elt].data_ptr<scalar_t>();
                    const scalar_t *data_lut1d = lut1d[elt].data_ptr<scalar_t>();
                    scalar_t *data_col = output[elt].data_ptr<scalar_t>();

                    seplut_transform_cpu_forward_impl(
                        num_kernels, data_inp, data_lut3d, data_lut1d,
                        height, width, num_channels, stride_lut3d, stride_lut1d,
                        data_col);
                }));
        }
    }
}