#include "tensor.hpp"

#include "../utils.hpp"

#include <cstring>
#include <numeric>
#include <sstream>

namespace llaisys {

Tensor::Tensor(TensorMeta meta, core::storage_t storage, size_t offset)
    : _meta(std::move(meta)), _storage(std::move(storage)), _offset(offset) {}

tensor_t Tensor::create(const std::vector<size_t> &shape,
                        llaisysDataType_t dtype,
                        llaisysDeviceType_t device_type,
                        int device) {
    size_t ndim_ = shape.size();
    std::vector<ptrdiff_t> strides(ndim_);
    size_t stride = 1;
    for (size_t i = 1; i <= ndim_; i++) {
        strides[ndim_ - i] = stride;
        stride *= shape[ndim_ - i];
    }
    TensorMeta meta{dtype, shape, strides};
    size_t total_elems = stride;
    size_t dtype_size = utils::dsize(dtype);

    if (device_type == LLAISYS_DEVICE_CPU && core::context().runtime().deviceType() != LLAISYS_DEVICE_CPU) {
        auto storage = core::context().runtime().allocateHostStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    } else {
        core::context().setDevice(device_type, device);
        auto storage = core::context().runtime().allocateDeviceStorage(total_elems * dtype_size);
        return std::shared_ptr<Tensor>(new Tensor(meta, storage));
    }
}

std::byte *Tensor::data() {
    return _storage->memory() + _offset;
}

const std::byte *Tensor::data() const {
    return _storage->memory() + _offset;
}

size_t Tensor::ndim() const {
    return _meta.shape.size();
}

const std::vector<size_t> &Tensor::shape() const {
    return _meta.shape;
}

const std::vector<ptrdiff_t> &Tensor::strides() const {
    return _meta.strides;
}

llaisysDataType_t Tensor::dtype() const {
    return _meta.dtype;
}

llaisysDeviceType_t Tensor::deviceType() const {
    return _storage->deviceType();
}

int Tensor::deviceId() const {
    return _storage->deviceId();
}

size_t Tensor::numel() const {
    return std::accumulate(_meta.shape.begin(), _meta.shape.end(), size_t(1), std::multiplies<size_t>());
}

size_t Tensor::elementSize() const {
    return utils::dsize(_meta.dtype);
}

std::string Tensor::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << this->dtype();

    return ss.str();
}

template <typename T>
void print_data(const T *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, size_t dim) {
    if (dim == shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            if constexpr (std::is_same_v<T, bf16_t> || std::is_same_v<T, fp16_t>) {
                std::cout << utils::cast<float>(data[i * strides[dim]]) << " ";
            } else {
                std::cout << data[i * strides[dim]] << " ";
            }
        }
        std::cout << std::endl;
    } else if (dim < shape.size() - 1) {
        for (size_t i = 0; i < shape[dim]; i++) {
            print_data(data + i * strides[dim], shape, strides, dim + 1);
        }
    }
}

void debug_print(const std::byte *data, const std::vector<size_t> &shape, const std::vector<ptrdiff_t> &strides, llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_BYTE:
        return print_data(reinterpret_cast<const char *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BOOL:
        return print_data(reinterpret_cast<const bool *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I8:
        return print_data(reinterpret_cast<const int8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I16:
        return print_data(reinterpret_cast<const int16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I32:
        return print_data(reinterpret_cast<const int32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_I64:
        return print_data(reinterpret_cast<const int64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U8:
        return print_data(reinterpret_cast<const uint8_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U16:
        return print_data(reinterpret_cast<const uint16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U32:
        return print_data(reinterpret_cast<const uint32_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_U64:
        return print_data(reinterpret_cast<const uint64_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F16:
        return print_data(reinterpret_cast<const fp16_t *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F32:
        return print_data(reinterpret_cast<const float *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_F64:
        return print_data(reinterpret_cast<const double *>(data), shape, strides, 0);
    case LLAISYS_DTYPE_BF16:
        return print_data(reinterpret_cast<const bf16_t *>(data), shape, strides, 0);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

void Tensor::debug() const {
    core::context().setDevice(this->deviceType(), this->deviceId());
    core::context().runtime().api()->device_synchronize();
    std::cout << this->info() << std::endl;
    if (this->deviceType() == LLAISYS_DEVICE_CPU) {
        debug_print(this->data(), this->shape(), this->strides(), this->dtype());
    } else {
        auto tmp_tensor = create({this->_storage->size()}, this->dtype());
        core::context().runtime().api()->memcpy_sync(
            tmp_tensor->data(),
            this->data(),
            this->numel() * this->elementSize(),
            LLAISYS_MEMCPY_D2H);
        debug_print(tmp_tensor->data(), this->shape(), this->strides(), this->dtype());
    }
}

bool Tensor::isContiguous() const {
    const auto &shape = this->shape();
    const auto &strides = this->strides();
    int ndim = this->ndim();

    // 预期步长，从最内层开始，初始应为 1
    size_t expected_stride = 1;

    // 从最后一维向第一维倒序检查
    for (int i = ndim - 1; i >= 0; --i) {
        // 如果该维度大小为 0，通常认为是连续的（虽然没数据）
        if (shape[i] == 0) return true;

        // 如果该维度大小 > 1，步长必须符合预期
        if (shape[i] > 1) {
            if (static_cast<size_t>(strides[i]) != expected_stride) {
                return false;
            }
            // 更新下一层（更外层）的预期步长
            expected_stride *= shape[i];
        }
    }

    return true;
}

tensor_t Tensor::permute(const std::vector<size_t> &order) const {
    size_t n = this->ndim();
    
    // 1. 安全检查：输入的维度顺序必须和原维度数量一致
    if (order.size() != n) {
        throw std::invalid_argument("permute: order size does not match tensor ndim");
    }

    // 2. 准备新的形状和步长
    std::vector<size_t> new_shape(n);
    std::vector<ptrdiff_t> new_strides(n);
    
    // 这里的逻辑是：新张量的第 i 维，对应原张量的第 order[i] 维
    for (size_t i = 0; i < n; ++i) {
        size_t old_dim = order[i];
        if (old_dim >= n) {
            throw std::out_of_range("permute: invalid dimension index");
        }
        new_shape[i] = _meta.shape[old_dim];
        new_strides[i] = _meta.strides[old_dim];
    }

    // 3. 构造新的元数据
    TensorMeta new_meta{this->dtype(), new_shape, new_strides};

    // 4. 返回新张量，共享存储和偏移
    return std::shared_ptr<Tensor>(new Tensor(new_meta, this->_storage, this->_offset));
}

tensor_t Tensor::view(const std::vector<size_t> &shape) const {
    // 1. 验证元素总数是否匹配
    size_t new_numel = std::accumulate(shape.begin(), shape.end(), size_t(1), std::multiplies<size_t>());
    if (new_numel != this->numel()) {
        throw std::runtime_error("view: total number of elements must not change.");
    }

    // 2. 检查兼容性：只有连续张量才能简单地执行 view
    // 注意：如果作业要求更宽松，可以不强制检查这个，但通常 view 依赖连续性
    if (!this->isContiguous()) {
        throw std::runtime_error("view: tensor is not contiguous. Use contiguous() before view.");
    }

    // 3. 计算新形状下的步长 (Row-major)
    size_t ndim = shape.size();
    std::vector<ptrdiff_t> new_strides(ndim);
    size_t stride = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        new_strides[i] = stride;
        stride *= shape[i];
    }

    // 4. 构造新元数据
    TensorMeta new_meta{this->dtype(), shape, new_strides};

    // 5. 返回新张量，共享存储和偏移
    // 注意：这里调用私有构造函数，需要使用 std::shared_ptr 的构造方式
    return std::shared_ptr<Tensor>(new Tensor(new_meta, this->_storage, this->_offset));
}

tensor_t Tensor::slice(size_t dim, size_t start, size_t end) const {
    // 1. 合法性检查
    if (dim >= this->ndim()) {
        throw std::out_of_range("slice: dimension out of range");
    }
    if (start > end || end > _meta.shape[dim]) {
        throw std::out_of_range("slice: invalid start/end indices");
    }

    // 2. 准备新的形状 (只有被切的那一维变了)
    std::vector<size_t> new_shape = this->shape();
    new_shape[dim] = end - start;

    // 3. 计算新的字节偏移量
    // 公式：原偏移 + (起始索引 * 该维步长 * 元素字节大小)
    size_t new_offset = this->_offset + (start * this->strides()[dim] * this->elementSize());

    // 4. 构造新元数据 (步长完全复用原来的)
    TensorMeta new_meta{this->dtype(), new_shape, this->strides()};

    // 5. 返回新张量，注意它和原张量共享同一个 _storage
    return std::shared_ptr<Tensor>(new Tensor(new_meta, this->_storage, new_offset));
}

void Tensor::load(const void *src) {
    // 1. 设置当前的设备环境，防止把数据拷错地方
    core::context().setDevice(this->deviceType(), this->deviceId());

    // 2. 计算需要复制的总字节数
    size_t bytes = this->numel() * this->elementSize();

    // 3. 执行内存拷贝
    // 参数含义：目标地址, 源地址, 字节数, 拷贝方向(Host to Device)
    core::context().runtime().api()->memcpy_sync(
        this->data(), 
        src, 
        bytes, 
        LLAISYS_MEMCPY_H2D
    );
}

tensor_t Tensor::contiguous() const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::reshape(const std::vector<size_t> &shape) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

tensor_t Tensor::to(llaisysDeviceType_t device_type, int device) const {
    TO_BE_IMPLEMENTED();
    return std::shared_ptr<Tensor>(new Tensor(_meta, _storage));
}

} // namespace llaisys
