#!/bin/bash

exw_virtuals_configure ()
{
    local exw_script_dir=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
    local exw_dir=$(dirname ${exw_script_dir})
    local nvcc_wrapper=${EXAWIND_CUDA_WRAPPER:-${exw_dir}/bin/nvcc_wrapper}
    local extra_args="$@"

    local cmake_cmd=(
        cmake
        -DCMAKE_CXX_COMPILER=${nvcc_wrapper}
        -DCMAKE_CXX_STANDARD:STRING="14"
        -DKokkos_ENABLE_SERIAL=ON
        -DKokkos_ENABLE_OPENMP=OFF
        -DKokkos_ENABLE_CUDA=ON
        -DKokkos_ENABLE_HIP=OFF
        -DKokkos_ENABLE_OPENMPTARGET=OFF
        -DKokkos_ENABLE_CUDA_LAMBDA=ON
        -DKokkos_ENABLE_CUDA_RELOCATABLE_DEVICE_CODE=ON
        -DKokkos_ARCH_VOLTA70=ON
        ${extra_args}
        ${EXW_VIRTUALS_SOURCE_DIR:-..}
    )

    echo "${cmake_cmd[@]}" | tee cmake_output.log
    eval "${cmake_cmd[@]}" |& tee -a cmake_output.log
}

exw_virtuals_configure "$@"
