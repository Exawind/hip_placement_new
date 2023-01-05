#!/bin/bash
module load rocm/5.4.0
placement_new_configure ()
{
    local extra_args="$@"

    local cmake_cmd=(
        cmake
        -DCMAKE_CXX_COMPILER=$(which hipcc)
        -DCMAKE_C_COMPILER=$(which hipcc)
        -DCMAKE_CXX_STANDARD:STRING="17"
		  -DCMAKE_CXX_FLAGS="--gpu-max-threads-per-block=128"
        -DKokkos_ENABLE_SERIAL=ON
        -DKokkos_ENABLE_OPENMP=OFF
        -DKokkos_ENABLE_CUDA=OFF
        -DKokkos_ENABLE_HIP=ON
        -DKokkos_ENABLE_OPENMPTARGET=OFF
        -DKokkos_ARCH_VEGA90A=ON
        -DKokkos_ENABLE_HIP_RELOCATABLE_DEVICE_CODE=OFF
        ${extra_args}
        ${PLACEMENT_NEW_SOURCE_DIR:-..}
    )

    echo "${cmake_cmd[@]}" | tee cmake_output.log
    eval "${cmake_cmd[@]}" |& tee -a cmake_output.log
}

placement_new_configure "$@"
