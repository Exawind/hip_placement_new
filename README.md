```
# ON crusher
module load rocm/5.4.0

# Clone codebase
git clone --recursive https://github.com/exawind/hip_placement_new.git

# Create build directory
cd hip_placement_new

# Configure and build for host
mkdir build_hip
cd build_hip


##	Passing Test
# CMake configure step
../scripts/configure-hip.sh -DGPU_MAX_THREADS_PER_BLOCK=OFF

# compile
make -j 8

# Run executable on an interactive node
./exw_placement_new


##	Failing Test
# CMake configure step
../scripts/configure-hip.sh -DGPU_MAX_THREADS_PER_BLOCK=ON

This will compile ONLY main.cpp executable witht the flag --gpu-max-threads_per-block=128.
You can modify the value in src/CMakeLists.txt

# compile
make -j 8

# Run executable on an interactive node
./exw_placement_new
```