/*
 * Copyright 2018 Brett Witherspoon
 */

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>

#include <boost/program_options.hpp>
#include <boost/compute/core.hpp>

#include "fft.hpp"

namespace compute = boost::compute;
namespace po = boost::program_options;

int main(int argc, char *argv[]) {
    unsigned length;

    po::options_description desc("Supported options");
    desc.add_options()
        ("help,h", "print help message")
        ("length,l", po::value<decltype(length)>(&length)->default_value(8), "set FFT length")
        ("verbose,v", "print verbose messages");
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
        std::cerr << desc << std::endl;
        return 1;
    }

    compute::device device = compute::system::default_device();
    compute::context context(device);
    compute::command_queue queue(context, device, compute::command_queue::enable_profiling);

    // Print some device information
    std::cout << device.platform().name() << ": " << device.name() << std::endl;
    std::cout << "Global memory size: " << device.global_memory_size() << std::endl;
    std::cout << "Local memory size: " << device.local_memory_size() << std::endl;
    std::cout << "Compute units: " << device.compute_units() << std::endl;
    std::cout << "Preferred vector width: " <<device.preferred_vector_width<float>() << std::endl;

    // Create FFT object
    ocl::fft fft(queue, length);

    // Initialize input buffer
    auto input = fft.map(compute::command_queue::map_write);
    std::default_random_engine eng;
    std::normal_distribution<> dis {0, 1};
    auto rng = std::bind(dis, eng);
    std::generate(input, input + length, rng);
    if (vm.count("verbose")) {
        std::cout << "Input: " << std::endl;
        for (auto i = 0u; i < length; ++i)
            std::cout << input[i] << std::endl;
    }
    fft.unmap().wait();

    // Enqueue kernels
    auto events = fft();
    events.wait();

    // Print profiling information
    std::chrono::nanoseconds nanoseconds {0};
    for (const auto &event : events)
        nanoseconds += event.duration<std::chrono::nanoseconds>();
    std::cout << "Execute time: " << nanoseconds.count() << " ns" << std::endl;

    // Print output buffer
    auto output = fft.map(compute::command_queue::map_read);
    if (vm.count("verbose")){
        std::cout << "Output: " << std::endl;
        for (auto i = 0u; i < length; ++i) std::cout << output[i] << std::endl;
    }
    fft.unmap().wait();

    return 0;
}
