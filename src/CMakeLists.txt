add_executable(benchmark benchmark.cpp ${PROJECT_SOURCE_DIR}/src/fft.cpp)
target_include_directories(benchmark PRIVATE ${Boost_INCLUDE_DIRS})
target_link_libraries(benchmark Boost::program_options OpenCL::OpenCL)
