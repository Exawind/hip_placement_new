# ON crusher
module load rocm/5.4.0

# Clone codebase
git clone --recurse-submodules https://github.com/exawind/hip_placement_new.git

# Create build directory
cd hip_placement_new

# Configure and build for host
mkdir build_hip
cd build_hip

# CMake configure step
../scripts/configure-hip.sh 

# compile
make -j 8

# Run executable on an interactive node
./exw_placement_new
