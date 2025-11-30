#include "ggml.h"
#include "ggml-cpu.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int ggml_linear_forward(const float * input,
                        const float * weight,
                        const float * bias,
                        float * output,
                        int batch,
                        int in_features,
                        int out_features,
                        int n_threads) {
    size_t tensor_space = (size_t)(batch * in_features + out_features * in_features + batch * out_features);
    size_t ctx_size = tensor_space * sizeof(float) * 4 + ggml_tensor_overhead() * 8;
    struct ggml_init_params params = {
        .mem_size   = ctx_size,
        .mem_buffer = NULL,
        .no_alloc   = false,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return -1;
    }

    struct ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_features, batch);
    struct ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_features, out_features);
    struct ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, out_features);
    memcpy(x->data, input, (size_t)batch * in_features * sizeof(float));
    memcpy(w->data, weight, (size_t)out_features * in_features * sizeof(float));
    memcpy(b->data, bias, (size_t)out_features * sizeof(float));

    struct ggml_tensor * y = ggml_mul_mat(ctx, w, x);
    struct ggml_tensor * br = ggml_repeat(ctx, b, y);
    struct ggml_tensor * out = ggml_add(ctx, y, br);

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    ggml_graph_compute_with_ctx(ctx, gf, n_threads);

    memcpy(output, ggml_get_data_f32(out), (size_t)batch * out_features * sizeof(float));
    ggml_free(ctx);
    return 0;
}

#ifdef __cplusplus
}
#endif
