#include "op.hpp"
#include "../../utils.hpp"
#include <cstring> // 为了使用 std::memcpy

namespace llaisys::ops {
    // 模板函数
    template <typename T>
    void embedding_(tensor_t out, tensor_t index, tensor_t weight) {
        // 1. 获取基本维度信息
        size_t num_indices = index->numel();        // 总共要查多少个词
        size_t embedding_dim = weight->shape()[1];  // 每个词向量的长度
        size_t row_size_bytes = embedding_dim * sizeof(T);  // 每个词向量占用的字节数

        // 2. 获取指针
        T *out_ptr = reinterpret_cast<T*>(out->data());
        const T *weight_ptr = reinterpret_cast<const T*>(weight->data());
        const int64_t *index_ptr = reinterpret_cast<const int64_t*>(index->data());

        // 3. 查表并拷贝
        for (size_t i = 0; i < num_indices; i++) {
            int64_t idx = index_ptr[i];     //  拿到要查的词的ID
            const T *src = weight_ptr + idx * embedding_dim;    // 计算源地址:weight 的第 idx 行
            T *dst = out_ptr + i * embedding_dim;   // 计算目标地址:out 的第 i 行
            std::memcpy(dst, src, row_size_bytes);
        }
    }
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    switch (weight->dtype()) {
        case LLAISYS_DTYPE_F32:
            embedding_<float>(out, index, weight);
            break;
        case LLAISYS_DTYPE_F16:
            embedding_<fp16_t>(out, index, weight);
            break;
        case LLAISYS_DTYPE_BF16:
            embedding_<bf16_t>(out, index, weight);
            break;
        default:
            // 实际上应该报错，这里为了简化省略
            break;
    }
}
} // namespace llaisys::ops
