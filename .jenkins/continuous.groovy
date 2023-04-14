pipeline {
    options {
        disableConcurrentBuilds(abortPrevious: true)
        timeout(time: 3, unit: 'HOURS')
    }
    triggers {
        issueCommentTrigger('.*test this please.*')
    }
    agent none

    environment {
        CCACHE_DIR = '/tmp/ccache'
        CCACHE_MAXSIZE = '10G'
        ARBORX_DIR = '/opt/arborx'
        BENCHMARK_COLOR = 'no'
        BOOST_TEST_COLOR_OUTPUT = 'no'
        CTEST_OPTIONS = '--timeout 180 --no-compress-output -T Test --test-output-size-passed=65536 --test-output-size-failed=1048576'
        OMP_NUM_THREADS = 8
        OMP_PLACES = 'threads'
        OMP_PROC_BIND = 'spread'
    }
    stages {

        stage("Style") {
            agent {
                dockerfile {
                    filename "Dockerfile.clang-format"
                    dir "docker"
                    additionalBuildArgs "--build-arg CLANG_FORMAT_VERSION=14.0.0"
                    label 'docker'
                }
            }
            steps {
                sh './scripts/check_format_cpp.sh'
            }
        }

        stage('Build') {
            parallel {
                stage('CUDA-11.5.2-NVCC-CUDA-AWARE-MPI') {
                    agent {
                        dockerfile {
                            filename "Dockerfile"
                            dir "docker"
                            additionalBuildArgs '--build-arg BASE=nvidia/cuda:11.5.2-devel-ubuntu20.04 --build-arg KOKKOS_VERSION=3.7.01 --build-arg KOKKOS_OPTIONS="-DCMAKE_CXX_EXTENSIONS=OFF -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON -DKokkos_ARCH_VOLTA70=ON" --build-arg CUDA_AWARE_MPI=1'
                            args '-v /tmp/ccache:/tmp/ccache --env NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES}'
                            label 'NVIDIA_Tesla_V100-PCIE-32GB && nvidia-docker'
                        }
                    }
                    steps {
                        sh 'ccache --zero-stats'
                        sh 'rm -rf build && mkdir -p build'
                        dir('build') {
                            sh '''
                                cmake \
                                    -D CMAKE_INSTALL_PREFIX=$ARBORX_DIR \
                                    -D CMAKE_BUILD_TYPE=Debug \
                                    -D CMAKE_CXX_COMPILER_LAUNCHER=ccache \
                                    -D CMAKE_CXX_COMPILER=$KOKKOS_DIR/bin/nvcc_wrapper \
                                    -D CMAKE_CXX_EXTENSIONS=OFF \
                                    -D CMAKE_CXX_FLAGS="-Wpedantic -Wall -Wextra" \
                                    -D CMAKE_PREFIX_PATH="$KOKKOS_DIR;$BOOST_DIR;$BENCHMARK_DIR" \
                                    -D ARBORX_ENABLE_MPI=ON \
                                    -D MPIEXEC_PREFLAGS="--allow-run-as-root" \
                                    -D MPIEXEC_MAX_NUMPROCS=4 \
                                    -D ARBORX_ENABLE_GPU_AWARE_MPI=ON \
                                    -D ARBORX_ENABLE_TESTS=ON \
                                    -D ARBORX_ENABLE_EXAMPLES=ON \
                                    -D ARBORX_ENABLE_BENCHMARKS=ON \
                                ..
                            '''
                            sh 'make -j8 VERBOSE=1'
                            sh 'ctest $CTEST_OPTIONS'
                        }
                    }
                    post {
                        always {
                            sh 'ccache --show-stats'
                            xunit reduceLog: false, tools:[CTest(deleteOutputFiles: true, failIfNotNew: true, pattern: 'build/Testing/**/Test.xml', skipNoTestFiles: false, stopProcessingIfError: true)]
                        }
                        success {
                            sh 'cd build && make install'
                            sh 'rm -rf test_install && mkdir -p test_install'
                            dir('test_install') {
                                sh 'cp -r ../examples .'
                                sh '''
                                    cmake \
                                        -D CMAKE_CXX_COMPILER_LAUNCHER=ccache \
                                        -D CMAKE_CXX_COMPILER=$KOKKOS_DIR/bin/nvcc_wrapper \
                                        -D CMAKE_CXX_EXTENSIONS=OFF \
                                        -D CMAKE_PREFIX_PATH="$KOKKOS_DIR;$ARBORX_DIR" \
                                    examples \
                                '''
                                sh 'make VERBOSE=1'
                                sh 'make test'
                            }
                        }
                    }
                }
                stage('CUDA-11.0.3-NVCC') {
                    agent {
                        dockerfile {
                            filename "Dockerfile"
                            dir "docker"
                            additionalBuildArgs '--build-arg BASE=nvidia/cuda:11.0.3-devel-ubuntu20.04 --build-arg KOKKOS_OPTIONS="-DCMAKE_CXX_EXTENSIONS=OFF -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON -DKokkos_ARCH_VOLTA70=ON"'
                            args '-v /tmp/ccache:/tmp/ccache --env NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES}'
                            label 'NVIDIA_Tesla_V100-PCIE-32GB && nvidia-docker'
                        }
                    }
                    steps {
                        sh 'ccache --zero-stats'
                        sh 'rm -rf build && mkdir -p build'
                        dir('build') {
                            sh '''
                                cmake \
                                    -D CMAKE_INSTALL_PREFIX=$ARBORX_DIR \
                                    -D CMAKE_BUILD_TYPE=Debug \
                                    -D CMAKE_CXX_COMPILER_LAUNCHER=ccache \
                                    -D CMAKE_CXX_COMPILER=$KOKKOS_DIR/bin/nvcc_wrapper \
                                    -D CMAKE_CXX_EXTENSIONS=OFF \
                                    -D CMAKE_CXX_FLAGS="-Wpedantic -Wall -Wextra" \
                                    -D CMAKE_PREFIX_PATH="$KOKKOS_DIR;$BOOST_DIR;$BENCHMARK_DIR" \
                                    -D ARBORX_ENABLE_MPI=ON \
                                    -D MPIEXEC_PREFLAGS="--allow-run-as-root" \
                                    -D MPIEXEC_MAX_NUMPROCS=4 \
                                    -D ARBORX_ENABLE_TESTS=ON \
                                    -D ARBORX_ENABLE_EXAMPLES=ON \
                                    -D ARBORX_ENABLE_BENCHMARKS=ON \
                                ..
                            '''
                            sh 'make -j8 VERBOSE=1'
                            sh 'ctest $CTEST_OPTIONS'
                        }
                    }
                    post {
                        always {
                            sh 'ccache --show-stats'
                            xunit reduceLog: false, tools:[CTest(deleteOutputFiles: true, failIfNotNew: true, pattern: 'build/Testing/**/Test.xml', skipNoTestFiles: false, stopProcessingIfError: true)]
                        }
                        success {
                            sh 'cd build && make install'
                            sh 'rm -rf test_install && mkdir -p test_install'
                            dir('test_install') {
                                sh 'cp -r ../examples .'
                                sh '''
                                    cmake \
                                        -D CMAKE_CXX_COMPILER_LAUNCHER=ccache \
                                        -D CMAKE_CXX_COMPILER=$KOKKOS_DIR/bin/nvcc_wrapper \
                                        -D CMAKE_CXX_EXTENSIONS=OFF \
                                        -D CMAKE_PREFIX_PATH="$KOKKOS_DIR;$ARBORX_DIR" \
                                    examples \
                                '''
                                sh 'make VERBOSE=1'
                                sh 'make test'
                            }
                        }
                    }
                }
                stage('CUDA-11.0.3-Clang') {
                    agent {
                        dockerfile {
                            filename "Dockerfile"
                            dir "docker"
                            additionalBuildArgs '--build-arg BASE=nvidia/cuda:11.0.3-devel-ubuntu18.04 --build-arg KOKKOS_VERSION="3.7.01" --build-arg KOKKOS_OPTIONS="-DCMAKE_CXX_EXTENSIONS=OFF -DCMAKE_CXX_COMPILER=clang++ -DKokkos_ENABLE_THREADS=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON -DKokkos_ARCH_VOLTA70=ON -DCMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu"'
                            args '-v /tmp/ccache:/tmp/ccache --env NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES}'
                            label 'NVIDIA_Tesla_V100-PCIE-32GB && nvidia-docker'
                        }
                    }
                    steps {
                        sh 'ccache --zero-stats'
                        sh 'rm -rf build && mkdir -p build'
                        dir('build') {
                            sh '''
                                cmake \
                                    -D CMAKE_INSTALL_PREFIX=$ARBORX_DIR \
                                    -D CMAKE_BUILD_TYPE=Debug \
                                    -D CMAKE_CXX_COMPILER_LAUNCHER=ccache \
                                    -D CMAKE_CXX_COMPILER=clang++ \
                                    -D CMAKE_CXX_EXTENSIONS=OFF \
                                    -D CMAKE_CXX_FLAGS="-Wpedantic -Wall -Wextra" \
                                    -D CMAKE_PREFIX_PATH="$KOKKOS_DIR;$BOOST_DIR;$BENCHMARK_DIR" \
                                    -D ARBORX_ENABLE_MPI=OFF \
                                    -D ARBORX_ENABLE_TESTS=ON \
                                    -D ARBORX_ENABLE_EXAMPLES=ON \
                                    -D ARBORX_ENABLE_BENCHMARKS=ON \
                                ..
                            '''
                            sh 'make -j8 VERBOSE=1'
                            sh 'ctest $CTEST_OPTIONS'
                        }
                    }
                    post {
                        always {
                            sh 'ccache --show-stats'
                            xunit reduceLog: false, tools:[CTest(deleteOutputFiles: true, failIfNotNew: true, pattern: 'build/Testing/**/Test.xml', skipNoTestFiles: false, stopProcessingIfError: true)]
                        }
                        success {
                            sh 'cd build && make install'
                            sh 'rm -rf test_install && mkdir -p test_install'
                            dir('test_install') {
                                sh 'cp -r ../examples .'
                                sh '''
                                    cmake \
                                        -D CMAKE_CXX_COMPILER_LAUNCHER=ccache \
                                        -D CMAKE_CXX_COMPILER=clang++ \
                                        -D CMAKE_CXX_EXTENSIONS=OFF \
                                        -D CMAKE_PREFIX_PATH="$KOKKOS_DIR;$ARBORX_DIR" \
                                    examples \
                                '''
                                sh 'make VERBOSE=1'
                                sh 'make test'
                            }
                        }
                    }
                }

                stage('Clang') {
                    agent {
                        dockerfile {
                            filename "Dockerfile"
                            dir "docker"
                            additionalBuildArgs '--build-arg BASE=ubuntu:18.04 --build-arg KOKKOS_OPTIONS="-DCMAKE_CXX_EXTENSIONS=OFF -DKokkos_ENABLE_OPENMP=ON -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu"'
                            args '-v /tmp/ccache:/tmp/ccache'
                            label 'docker'
                        }
                    }
                    steps {
                        sh 'ccache --zero-stats'
                        sh 'rm -rf build && mkdir -p build'
                        dir('build') {
                            sh '''
                                cmake \
                                    -D CMAKE_INSTALL_PREFIX=$ARBORX_DIR \
                                    -D CMAKE_BUILD_TYPE=Debug \
                                    -D CMAKE_CXX_COMPILER_LAUNCHER=ccache \
                                    -D CMAKE_CXX_COMPILER=clang++ \
                                    -D CMAKE_CXX_EXTENSIONS=OFF \
                                    -D CMAKE_CXX_FLAGS="-Wpedantic -Wall -Wextra" \
                                    -D CMAKE_CXX_CLANG_TIDY="$LLVM_DIR/bin/clang-tidy" \
                                    -D CMAKE_PREFIX_PATH="$KOKKOS_DIR;$BOOST_DIR;$BENCHMARK_DIR" \
                                    -D ARBORX_ENABLE_MPI=ON \
                                    -D MPIEXEC_PREFLAGS="--allow-run-as-root" \
                                    -D MPIEXEC_MAX_NUMPROCS=4 \
                                    -D ARBORX_ENABLE_TESTS=ON \
                                    -D ARBORX_ENABLE_EXAMPLES=ON \
                                    -D ARBORX_ENABLE_BENCHMARKS=ON \
                                ..
                            '''
                            sh 'make -j6 VERBOSE=1'
                            sh 'ctest $CTEST_OPTIONS'
                        }
                    }
                    post {
                        always {
                            sh 'ccache --show-stats'
                            xunit reduceLog: false, tools:[CTest(deleteOutputFiles: true, failIfNotNew: true, pattern: 'build/Testing/**/Test.xml', skipNoTestFiles: false, stopProcessingIfError: true)]
                        }
                        success {
                            sh 'cd build && make install'
                            sh 'rm -rf test_install && mkdir -p test_install'
                            dir('test_install') {
                                sh 'cp -r ../examples .'
                                sh '''
                                    cmake \
                                        -D CMAKE_CXX_COMPILER_LAUNCHER=ccache \
                                        -D CMAKE_CXX_COMPILER=clang++ \
                                        -D CMAKE_CXX_EXTENSIONS=OFF \
                                        -D CMAKE_PREFIX_PATH="$KOKKOS_DIR;$ARBORX_DIR" \
                                    examples \
                                '''
                                sh 'make VERBOSE=1'
                                sh 'make test'
                            }
                        }
                    }
                }

                stage('GCC-12.2') {
                    agent {
                        dockerfile {
                            filename "Dockerfile"
                            dir "docker"
                            additionalBuildArgs '--build-arg BASE=gcc:12.2.0 --build-arg KOKKOS_OPTIONS="-DCMAKE_CXX_EXTENSIONS=OFF -DKokkos_ENABLE_OPENMP=ON -DCMAKE_CXX_COMPILER=g++ -DCMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu"'
                            args '-v /tmp/ccache:/tmp/ccache'
                            label 'docker'
                        }
                    }
                    steps {
                        sh 'ccache --zero-stats'
                        sh 'rm -rf build && mkdir -p build'
                        dir('build') {
                            sh '''
                                cmake \
                                    -D CMAKE_INSTALL_PREFIX=$ARBORX_DIR \
                                    -D CMAKE_BUILD_TYPE=Debug \
                                    -D CMAKE_CXX_COMPILER_LAUNCHER=ccache \
                                    -D CMAKE_CXX_COMPILER=g++ \
                                    -D CMAKE_CXX_EXTENSIONS=OFF \
                                    -D CMAKE_CXX_FLAGS="-Wpedantic -Wall -Wextra" \
                                    -D CMAKE_PREFIX_PATH="$KOKKOS_DIR;$BOOST_DIR;$BENCHMARK_DIR" \
                                    -D ARBORX_ENABLE_MPI=ON \
                                    -D MPIEXEC_PREFLAGS="--allow-run-as-root" \
                                    -D MPIEXEC_MAX_NUMPROCS=4 \
                                    -D ARBORX_ENABLE_TESTS=ON \
                                    -D ARBORX_ENABLE_EXAMPLES=ON \
                                    -D ARBORX_ENABLE_BENCHMARKS=ON \
                                ..
                            '''
                            sh 'make -j6 VERBOSE=1'
                            sh 'ctest $CTEST_OPTIONS'
                        }
                    }
                    post {
                        always {
                            sh 'ccache --show-stats'
                            xunit reduceLog: false, tools:[CTest(deleteOutputFiles: true, failIfNotNew: true, pattern: 'build/Testing/**/Test.xml', skipNoTestFiles: false, stopProcessingIfError: true)]
                        }
                        success {
                            sh 'cd build && make install'
                            sh 'rm -rf test_install && mkdir -p test_install'
                            dir('test_install') {
                                sh 'cp -r ../examples .'
                                sh '''
                                    cmake \
                                        -D CMAKE_CXX_COMPILER_LAUNCHER=ccache \
                                        -D CMAKE_CXX_COMPILER=g++ \
                                        -D CMAKE_CXX_EXTENSIONS=OFF \
                                        -D CMAKE_PREFIX_PATH="$KOKKOS_DIR;$ARBORX_DIR" \
                                    examples \
                                '''
                                sh 'make VERBOSE=1'
                                sh 'make test'
                            }
                        }
                    }
                }

                stage('HIP-5.3.3') {
                    agent {
                        dockerfile {
                            filename "Dockerfile.hipcc"
                            dir "docker"
                            additionalBuildArgs '--build-arg BASE=rocm/dev-ubuntu-20.04:5.3.3-complete --build-arg KOKKOS_ARCH=${KOKKOS_ARCH}'
                            args '-v /tmp/ccache.kokkos:/tmp/ccache --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --group-add video --env HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES} --env AMDGPU_TARGET=${AMDGPU_TARGET}'
                            label 'rocm-docker && vega'
                        }
                    }
                    steps {
                        sh 'ccache --zero-stats'
                        sh 'rm -rf build && mkdir -p build'
                        dir('build') {
                            sh '''
                                cmake \
                                    -D CMAKE_INSTALL_PREFIX=$ARBORX_DIR \
                                    -D CMAKE_BUILD_TYPE=Debug \
                                    -D CMAKE_CXX_COMPILER=hipcc \
                                    -D CMAKE_CXX_EXTENSIONS=OFF \
                                    -D CMAKE_CXX_FLAGS="-DNDEBUG -Wpedantic -Wall -Wextra" \
                                    -D CMAKE_PREFIX_PATH="$KOKKOS_DIR;$BOOST_DIR;$BENCHMARK_DIR" \
                                    -D ARBORX_ENABLE_MPI=ON \
                                    -D MPIEXEC_PREFLAGS="--allow-run-as-root" \
                                    -D MPIEXEC_MAX_NUMPROCS=4 \
                                    -D CMAKE_EXE_LINKER_FLAGS="-lopen-pal" \
                                    -D GPU_TARGETS=${AMDGPU_TARGET} \
                                    -D ARBORX_ENABLE_TESTS=ON \
                                    -D ARBORX_ENABLE_EXAMPLES=ON \
                                    -D ARBORX_ENABLE_BENCHMARKS=ON \
                                ..
                            '''
                            sh 'make -j8 VERBOSE=1'
                            sh 'ctest $CTEST_OPTIONS'
                        }
                    }
                    post {
                        always {
                            sh 'ccache --show-stats'
                            xunit reduceLog: false, tools:[CTest(deleteOutputFiles: true, failIfNotNew: true, pattern: 'build/Testing/**/Test.xml', skipNoTestFiles: false, stopProcessingIfError: true)]
                        }
                        success {
                            sh 'cd build && make install'
                            sh 'rm -rf test_install && mkdir -p test_install'
                            dir('test_install') {
                                sh 'cp -r ../examples .'
                                sh '''
                                    cmake \
                                        -D CMAKE_EXE_LINKER_FLAGS="-lopen-pal" \
                                        -D GPU_TARGETS=${AMDGPU_TARGET} \
                                        -D CMAKE_CXX_COMPILER=hipcc \
                                        -D CMAKE_CXX_EXTENSIONS=OFF \
                                        -D CMAKE_BUILD_TYPE=RelWithDebInfo \
                                        -D CMAKE_PREFIX_PATH="$KOKKOS_DIR;$ARBORX_DIR" \
                                    examples \
                                '''
                                sh 'make VERBOSE=1'
                                sh 'ctest --output-on-failure'
                            }
                        }
                    }
                }

                stage('SYCL') {
                    agent {
                        dockerfile {
                            filename "Dockerfile.sycl"
                            dir "docker"
                            args '-v /tmp/ccache.kokkos:/tmp/ccache'
                            label 'NVIDIA_Tesla_V100-PCIE-32GB && nvidia-docker'
                        }
                    }
                    steps {
                        sh 'ccache --zero-stats'
                        sh 'rm -rf build && mkdir -p build'
                        dir('build') {
                            sh '''
                                cmake \
                                    -D CMAKE_INSTALL_PREFIX=$ARBORX_DIR \
                                    -D CMAKE_BUILD_TYPE=Release \
                                    -D CMAKE_CXX_COMPILER=clang++ \
                                    -D CMAKE_CXX_EXTENSIONS=OFF \
                                    -D CMAKE_CXX_FLAGS="-fsycl-device-code-split=per_kernel -Wpedantic -Wall -Wextra -Wno-unknown-cuda-version" \
                                    -D CMAKE_PREFIX_PATH="$KOKKOS_DIR;$BOOST_DIR;$BENCHMARK_DIR;$ONE_DPL_DIR" \
                                    -D ARBORX_ENABLE_MPI=ON \
                                    -D MPIEXEC_PREFLAGS="--allow-run-as-root" \
                                    -D MPIEXEC_MAX_NUMPROCS=4 \
                                    -D ARBORX_ENABLE_TESTS=ON \
                                    -D ARBORX_ENABLE_EXAMPLES=ON \
                                    -D ARBORX_ENABLE_BENCHMARKS=ON \
                                    -D ARBORX_ENABLE_ONEDPL=ON \
                                    -D ONEDPL_PAR_BACKEND=serial \
                                ..
                            '''
                            sh 'make -j8 VERBOSE=1'
                            sh 'ctest $CTEST_OPTIONS'
                        }
                    }
                    post {
                        always {
                            sh 'ccache --show-stats'
                            xunit reduceLog: false, tools:[CTest(deleteOutputFiles: true, failIfNotNew: true, pattern: 'build/Testing/**/Test.xml', skipNoTestFiles: false, stopProcessingIfError: true)]
                        }
                        success {
                            sh 'cd build && make install'
                            sh 'rm -rf test_install && mkdir -p test_install'
                            dir('test_install') {
                                sh 'cp -r ../examples .'
                                sh '''
                                    cmake \
                                        -D CMAKE_BUILD_TYPE=Release \
                                        -D CMAKE_CXX_COMPILER=clang++ \
                                        -D CMAKE_CXX_EXTENSIONS=OFF \
                                        -D CMAKE_CXX_FLAGS="-Wno-unknown-cuda-version" \
                                        -D CMAKE_PREFIX_PATH="$KOKKOS_DIR;$ARBORX_DIR;$ONE_DPL_DIR" \
                                        -D ONEDPL_PAR_BACKEND=serial \
                                    examples \
                                '''
                                sh 'make VERBOSE=1'
                                sh 'make test'
                            }
                        }
                    }
                }
            }
        }
    }
    post {
        always {
            node('docker') {
                recordIssues(
                    enabledForFailure: true,
                    tools: [cmake(), gcc(), clang(), clangTidy()],
                    qualityGates: [[threshold: 1, type: 'TOTAL', unstable: true]],
                    filters: [excludeFile('/usr/local/cuda.*'), excludeCategory('#pragma-messages')]
                )
            }
        }
    }
}
