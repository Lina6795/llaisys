#pragma once
#include "../core/llaisys_core.hpp"

#include <vector>
namespace llaisys {
class Tensor;
using tensor_t = std::shared_ptr<Tensor>;   // 一个指向 Tensor 的智能指针

struct TensorMeta {
    llaisysDataType_t dtype;    // 数据类型，比如 float32
    std::vector<size_t> shape;  // 形状，比如 [2, 3, 4] 表示 2×3×4 的张量
    std::vector<ptrdiff_t> strides;  // 步长，比如 [12, 4, 1] 表示在内存中每个元素的偏移量
};

class Tensor {
private:
    TensorMeta _meta;   // 元信息：dtype, shape, strides
    core::storage_t _storage;   // 实际数据存在这里（可能是 shared_ptr<void> 或类似）
    size_t _offset; // 数据在 storage 中的起始偏移（用于切片等操作）
    Tensor(TensorMeta meta, core::storage_t storage, size_t offset = 0);

public:
    static tensor_t create(
        const std::vector<size_t> &shape,
        llaisysDataType_t dtype,
        llaisysDeviceType_t device_type = LLAISYS_DEVICE_CPU,
        int device = 0);
    ~Tensor() = default;
    // Info
    std::byte *data();
    const std::byte *data() const;
    size_t ndim() const;
    const std::vector<size_t> &shape() const;
    const std::vector<ptrdiff_t> &strides() const;
    llaisysDataType_t dtype() const;
    llaisysDeviceType_t deviceType() const;
    int deviceId() const;
    size_t numel() const;
    size_t elementSize() const;

    std::string info() const;
    void debug() const;

    bool isContiguous() const;

    // Meta Transform
    tensor_t permute(const std::vector<size_t> &order) const;
    tensor_t slice(size_t dim, size_t start, size_t end) const;
    tensor_t view(const std::vector<size_t> &shape) const;

    // Load data from host memory
    void load(const void *src);

    // Challenging features
    tensor_t contiguous() const;
    tensor_t reshape(const std::vector<size_t> &shape) const;
    tensor_t to(llaisysDeviceType_t device_type, int device = -1) const;
};

} // namespace llaisys
