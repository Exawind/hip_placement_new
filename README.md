```
# ON crusher
module load rocm/5.4.0

# Clone codebase
git clone --recursive https://github.com/exawind/hip_placement_new.git

# Create build directory
cd hip_placement_new

# Configure and build for hip
mkdir build_hip
cd build_hip


##	Passing Test
#  Open up configure-hip.sh, make sure that the line including
# -DCMAKE_CXX_FLAGS="--gpu-max-threads-per-block=128"  is commented out with #
../scripts/configure-hip.sh

# compile
make -j 8

# Run executable on an interactive node
./hip_placement_new


##	Failing Test
#  Open up configure-hip.sh, uncomment the line including
# -DCMAKE_CXX_FLAGS="--gpu-max-threads-per-block=128"
../scripts/configure-hip.sh 

# compile
make -j 8

# Run executable on an interactive node
./hip_placement_new
```