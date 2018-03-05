/*
 * Copyright 2018 Brett Witherspoon
 */

#include <cmath>
#include <iostream>
#include <stdexcept>

#include <boost/preprocessor/stringize.hpp>

#include "fft.hpp"

namespace {

const char *source = BOOST_PP_STRINGIZE(

    __kernel void window(__global float4 *x, uint n) {
      uint i = get_global_id(0);

      // Hamming
      float4 w;
      w.xy = 0.54 - 0.46 * cos(2 * M_PI_F * (2 * i) / (n - 1));
      w.zw = 0.54 - 0.46 * cos(2 * M_PI_F * (2 * i + 1) / (n - 1));

      x[i] = w * x[i];
    }

    __kernel void twiddle(__global float2 *w, uint n) {
      uint k = get_global_id(0);

      // e^{-j 2 \pi k/n)
      w[k] = cos((float2)(2 * M_PI_F * k / n, 2 * M_PI_F * k / n + M_PI_F / 2));
    }

    __kernel void reorder(__global float2 *x, uint s) {
      uint i = get_global_id(0);

      // Reverse bits
      uint j = i;
      j = (j & 0xFFFF0000) >> 16 | (j & 0x0000FFFF) << 16;
      j = (j & 0xFF00FF00) >> 8 | (j & 0x00FF00FF) << 8;
      j = (j & 0xF0F0F0F0) >> 4 | (j & 0x0F0F0F0F) << 4;
      j = (j & 0xCCCCCCCC) >> 2 | (j & 0x33333333) << 2;
      j = (j & 0xAAAAAAAA) >> 1 | (j & 0x55555555) << 1;

      // Adjust number of bits for log2(length)
      j >>= 32 - s;

      // Swap items
      if (i < j) {
        float2 tmp = x[i];
        x[i] = x[j];
        x[j] = tmp;
      }
    }

    __kernel void stage1(__global float4 *v) {
      uint i = get_global_id(0);

      // Butterfly
      float2 tmp = v[i].xy;
      v[i].xy = v[i].xy + v[i].zw;
      v[i].zw = tmp - v[i].zw;
    }

    __kernel void stagex(__global float2 *v, uint n, uint m, uint s) {
      // Indices
      uint i = get_global_id(0);
      uint j = 1 << (s - 1);

      // Twiddle
      uint mask = ~(0xFFFFFFFF << (s - 1));
      uint k = (i & mask) * (1 << (m - s));
      float2 w =
          cos((float2)(2 * M_PI_F * k / n, 2 * M_PI_F * k / n + M_PI_F / 2));

      // Complex multiply
      float2 z;
      z.x = v[i + j].x * w.x - v[i + j].y * w.y;
      z.y = v[i + j].x * w.y + v[i + j].y * w.x;

      // Butterfly
      v[i + j] = v[i] - z;
      v[i] = v[i] + z;
    });

inline bool ispow2(unsigned x) { return !(x == 0 || (x & (x - 1))); }

}  // namespace

namespace ocl {

fft::fft(boost::compute::command_queue &queue, unsigned length)
    : mapped_(nullptr),
      length_(length),
      stages_(static_cast<decltype(stages_)>(std::log2(length))),
      queue_(queue) {
  namespace compute = boost::compute;

  if (!ispow2(length))
    throw std::invalid_argument("Length must be a power of two");

  // Build program
  program_ = compute::program::create_with_source(source, queue_.get_context());

  // Create kernels
  try {
    program_.build();
    kernels_.push_back(program_.create_kernel("reorder"));
    kernels_.push_back(program_.create_kernel("stage1"));
    kernels_.push_back(program_.create_kernel("stagex"));
  } catch (compute::opencl_error &error) {
    // FIXME throw new error with build log
    std::cerr << error.what() << ": " << std::endl << program_.build_log();
    throw;
  }

  // Allocate buffers
  const auto flags =
      compute::buffer::read_write | compute::buffer::alloc_host_ptr;
  const auto size = length * sizeof(std::complex<float>);
  buffer_ = compute::buffer(queue_.get_context(), size, flags);

  // Set kernel arguments
  kernels_[0].set_arg(0, buffer_);
  kernels_[0].set_arg(1, stages_);

  kernels_[1].set_arg(0, buffer_);

  kernels_[2].set_arg(0, buffer_);
  kernels_[2].set_arg(1, length_);
  kernels_[2].set_arg(2, stages_);
  kernels_[2].set_arg(3, 1);
}

std::complex<float> *fft::map(cl_mem_flags flags,
                              const boost::compute::wait_list &events) {
  namespace compute = boost::compute;

  if (mapped_ != nullptr) return static_cast<std::complex<float> *>(mapped_);

  mapped_ =
      queue_.enqueue_map_buffer(buffer_, flags, 0, buffer_.size(), events);

  return static_cast<std::complex<float> *>(mapped_);
}

boost::compute::event fft::unmap() {
  if (mapped_ != nullptr) {
    auto event = queue_.enqueue_unmap_buffer(buffer_, mapped_);
    mapped_ = nullptr;
    return std::move(event);
  } else {
    return boost::compute::event();
  }
}

boost::compute::wait_list fft::operator()() {
  if (mapped_ != nullptr)
    throw std::runtime_error("OpenCL buffer still mapped to host");

  events_.clear();

  events_.insert(queue_.enqueue_1d_range_kernel(kernels_[0], 0, length_, 0));
  events_.insert(queue_.enqueue_1d_range_kernel(kernels_[1], 0, length_ / 2, 0,
                                                events_[0]));

  for (auto i = 2u; i <= stages_; ++i) {
    kernels_[2].set_arg(3, i);

    auto groups = length_ >> i;
    auto items = 1 << i;

    for (auto g = 0u; g < groups; ++g)
      events_.insert(queue_.enqueue_1d_range_kernel(
          kernels_[2], g * items, items / 2, 0, events_[i - 1]));
  }

  return events_;
}

}  // namespace ocl
