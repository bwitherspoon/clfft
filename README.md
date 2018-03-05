# OCL-FFT

An OpenCL implementation of the fast Fourier transform (FFT) algorithm optimized
for shared memory architectures.

## Dependencies

- [OpenCL](https://www.khronos.org/opencl/)
- [Boost.Compute](https://github.com/boostorg/compute)
- [Boost.Program_options](https://github.com/boostorg/program_options)

An Installable Client Driver (ICD) loader and an OpenCL implementation is
required. An open-source ICD loader is available for most Linux distributions.
An appropriate OpenCL implementation for your hardware is also required.
See [POCL](http://pocl.sourceforge.net/) for an open-source option.

### Fedora

```
dnf install ocl-icd-devel pocl
```
