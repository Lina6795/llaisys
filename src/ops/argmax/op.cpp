#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

namespace llaisys::ops {

    // 定义一个模板函数，同时处理float，fp16,bf16
    template <typename T>
    void argmax_(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
        // 1. 获取元素总数
        size_t count = vals->numel();
        // 2. 获取数据指针
        // 将 vals 张量的原始数据指针强转为 const T* 类型，以便后续按 T 类型读取元素
        const T* vals_data = reinterpret_cast<const T*>(vals->data());
        // 3. 初始化：最大值设为负无穷，索引设置为0
        float max_v = -std::numeric_limits<float>::infinity();
        int64_t max_i = 0;
        // 4. 遍历所有元素
        for (size_t i = 0; i < count; i++) {
            // 将当前元素转换为 float 类型，进行比较，避免半精度差
            float v = llaisys::utils::cast<float>(vals_data[i]);
            if (v > max_v) {
                max_v = v;
                max_i = i;
            }
        }
        // 5. 保存结果：最大值的索引 (int64)
        // max_idx 本身是一个 Tensor（shape 是 (1,) ，dtype 是 int64）
        int64_t* idx_ptr = reinterpret_cast<int64_t*>(max_idx->data());
        *idx_ptr = max_i;

        // 6. 保存结果：最大值本身 (类型要还原回去)
        T* val_ptr = reinterpret_cast<T*>(max_val->data());
        *val_ptr = utils::cast<T>(max_v);
    }
    // 主函数入口
    void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
        // 根据数据类型调用对应的模板函数
        switch (vals->dtype()) {
        case LLAISYS_DTYPE_F32:
            argmax_<float>(max_idx, max_val, vals);
            break;
        case LLAISYS_DTYPE_BF16:
            argmax_<llaisys::bf16_t>(max_idx, max_val, vals);
            break;
        case LLAISYS_DTYPE_F16:
            argmax_<llaisys::fp16_t>(max_idx, max_val, vals);
            break;
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(vals->dtype());
        }
    }
} // namespace llaisys::ops
