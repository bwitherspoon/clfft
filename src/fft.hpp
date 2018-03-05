/*
 * Copyright 2018 Brett Witherspoon
 */

#ifndef OCL_FFT_HPP_
#define OCL_FFT_HPP_

#include <complex>

#include <boost/compute/command_queue.hpp>
#include <boost/compute/kernel.hpp>
#include <boost/compute/program.hpp>

namespace ocl {

//! Compute the DFT using the FFT algorithm with OpenCL.
class fft {
 public:
  fft(boost::compute::command_queue& queue, unsigned length);

  fft(const fft&) = delete;

  ~fft() = default;

  fft& operator=(const fft&) = delete;

  std::complex<float>* map(
      cl_mem_flags flags,
      const boost::compute::wait_list& events = boost::compute::wait_list());

  boost::compute::event unmap();

  boost::compute::wait_list operator()();

 private:
  void* mapped_;
  const unsigned length_;
  const unsigned stages_;
  boost::compute::command_queue queue_;
  boost::compute::program program_;
  std::vector<boost::compute::kernel> kernels_;
  boost::compute::buffer buffer_;
  boost::compute::wait_list events_;
};

}  // namespace ocl

#endif /* OCL_FFT_HPP_ */
