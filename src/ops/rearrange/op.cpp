#include "op.hpp"
#include "../../utils.hpp"
#include <cstring>

namespace llaisys::ops {
    // 辅助函数：递归搬运数据
    // dst: 目标（连续内存）的当前指针
    // src: 源（非连续内存）的当前指针
    // shape: 剩余维度的形状
    // strides: 源 Tensor 在这些维度上的步幅
    // dim: 当前正在处理第几维
    // element_size: 每个元素的字节数 (例如 float 是 4)
    void rearrange_recursive(std::byte* dst, const std::byte* src, 
                         const std::vector<size_t>& shape, 
                         const std::vector<ptrdiff_t>& strides, 
                         int dim, size_t element_size) {
    
    // 【判断】是不是剥到了洋葱的最核心？（也就是 Tensor 的最后一维）
    if (dim == (int)shape.size() - 1) {
        // 拿到最后一维的步幅（在旧仓库里，拿下一个零件要跨多远）
        ptrdiff_t stride = strides[dim];
        // 拿到最后一维的数量（这一排有多少个零件）
        size_t count = shape[dim];

        // 【优化】如果旧仓库里的零件也是挨着放的（步幅为 1）
        if (stride == 1) {
            // 直接用大铲子一铲子全部铲走，速度最快！
            std::memcpy(dst, src, count * element_size);
        } else {
            // 如果旧仓库里零件中间有空隙，只能用小夹子一个一个夹
            for (size_t i = 0; i < count; ++i) {
                // dst 是新仓库，必须挨着放（i * element_size）
                // src 是旧仓库，按步幅跳着找（i * stride * element_size）
                std::memcpy(dst + i * element_size, src + i * stride * element_size, element_size);
            }
        }
    } else {
        // 【递归】还没剥到核心，继续处理这一层的大箱子
        size_t count = shape[dim];      // 这一层有多少个大箱子
        ptrdiff_t stride = strides[dim]; // 在旧仓库里，找下一个大箱子要跳多远
        
        // 【计算】为了让新仓库整齐，我得算出这一层箱子里所有东西总共占多大空间
        size_t inner_size = element_size;
        for (size_t k = dim + 1; k < shape.size(); ++k) {
            inner_size *= shape[k]; // 后面所有维度的长度乘起来
        }

        // 循环处理每一个大箱子
        for (size_t i = 0; i < count; ++i) {
            rearrange_recursive(
                // 目标位置：在新仓库里紧凑排队，每个大箱子占 inner_size 空间
                dst + i * inner_size,           
                // 来源位置：在旧仓库里，按这一层的步幅跳着找箱子
                src + i * stride * element_size, 
                shape, strides, dim + 1, element_size
            );
        }
    }
}
    void rearrange(tensor_t out, tensor_t in) {
    // 1. 【取经】从输入 Tensor 里拿到它的形状（shape）和步幅（strides）
    auto& shape = in->shape();
    auto& strides = in->strides();
    
    // 2. 【量尺寸】每个数据元素到底占几个字节（float=4, half=2）
    size_t element_size = in->elementSize();

    // 3. 【拿地址】拿到新旧仓库的“大门地址”（内存首地址指针）
    // 使用 std::byte* 是为了方便按字节数数，不关心具体数据类型
    std::byte* out_ptr = out->data();
    const std::byte* in_ptr = in->data();

    // 4. 【分情况处理】
    if (shape.size() == 0) {
        // 如果 Tensor 只是一个孤零零的数字（0维）
        // 别废话，直接一铲子搬走完事
        std::memcpy(out_ptr, in_ptr, element_size);
    } else {
        // 如果是多维数组，派递归搬运工从第 0 维（最外层）开始搬
        rearrange_recursive(out_ptr, in_ptr, shape, strides, 0, element_size);
    }
}
} // namespace llaisys::ops
