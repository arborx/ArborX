pipeline {
    options {
        timeout(time: 3, unit: 'HOURS')
    }
    agent none

    stages {

        stage('Build') {
            parallel {
                stage('CUDA-11.7.1') {
                    agent {
                        docker {
                            image 'nvidia/cuda:11.7.1-devel-ubuntu22.04'
                            label 'NVIDIA_Tesla_V100-PCIE-32GB && nvidia-docker'
                        }
                    }
                    environment {
                        CTEST_OPTIONS = '--timeout 180 --no-compress-output -T Test'
                        CMAKE_OPTIONS = '-D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_STANDARD=17 -D CMAKE_CXX_EXTENSIONS=OFF'
                    }
                    steps {
                        sh 'apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y git cmake libboost-program-options-dev libboost-test-dev libbenchmark-dev'
                        sh 'rm -rf source* build* install*'
                        sh 'git clone https://github.com/kokkos/kokkos.git --branch develop --depth 1 source-kokkos'
                        dir('source-kokkos') {
                            sh 'git rev-parse --short HEAD'
                        }
                        sh 'cmake -S source-kokkos -B build-kokkos -D CMAKE_INSTALL_PREFIX=$PWD/install-kokkos $CMAKE_OPTIONS -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ENABLE_CUDA_LAMBDA=ON'
                        sh 'cmake --build build-kokkos --parallel 8'
                        sh 'cmake --install build-kokkos'
                        sh 'cmake -B build-arborx -D CMAKE_INSTALL_PREFIX=$PWD/install-arborx -D Kokkos_ROOT=$PWD/install-kokkos $CMAKE_OPTIONS -D ARBORX_ENABLE_BENCHMARKS=ON'
                        sh 'cmake --build build-arborx --parallel 8'
                        dir('build-arborx') {
                            sh 'ctest $CTEST_OPTIONS'
                        }
                        sh 'cmake --install build-arborx'
                        sh 'cmake -S examples -B build-examples -D ArborX_ROOT=$PWD/install-arborx -D Kokkos_ROOT=$PWD/install-kokkos $CMAKE_OPTIONS'
                        sh 'cmake --build build-examples --parallel 8'
                        dir('build-examples') {
                            sh 'ctest $CTEST_OPTIONS'
                        }
                    }
                    post {
                        always {
                            xunit reduceLog: false, tools:[CTest(deleteOutputFiles: true, failIfNotNew: true, pattern: 'build-*/Testing/**/Test.xml', skipNoTestFiles: false, stopProcessingIfError: true)]
                        }
                    }
                }
                stage('ROCm-5.4') {
                    agent {
                        docker {
                            image 'rocm/dev-ubuntu-22.04:5.4-complete'
                            label 'AMD_Radeon_Instinct_MI100 && rocm-docker'
                            args '--device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined --group-add video --env HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES}'
                        }
                    }
                    environment {
                        CTEST_OPTIONS = '--timeout 180 --no-compress-output -T Test'
                        CMAKE_OPTIONS = '-D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_STANDARD=17 -D CMAKE_CXX_EXTENSIONS=OFF -DCMAKE_CXX_COMPILER=hipcc'
                        ROCM_PATH = '/opt/rocm'
                    }
                    steps {
                        sh 'apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y git cmake libboost-program-options-dev libboost-test-dev libbenchmark-dev'
                        sh 'rm -rf source* build* install*'
                        sh 'git clone https://github.com/kokkos/kokkos.git --branch develop --depth 1 source-kokkos'
                        dir('source-kokkos') {
                            sh 'git rev-parse --short HEAD'
                        }
                        sh 'cmake -S source-kokkos -B build-kokkos -D CMAKE_INSTALL_PREFIX=$PWD/install-kokkos $CMAKE_OPTIONS -D Kokkos_ENABLE_HIP=ON -DKokkos_ARCH_VEGA908=ON'
                        sh 'cmake --build build-kokkos --parallel 8'
                        sh 'cmake --install build-kokkos'
                        sh 'cmake -B build-arborx -D CMAKE_INSTALL_PREFIX=$PWD/install-arborx -D Kokkos_ROOT=$PWD/install-kokkos $CMAKE_OPTIONS -D ARBORX_ENABLE_BENCHMARKS=ON -DCMAKE_PREFIX_PATH=/opt/rocm'
                        sh 'cmake --build build-arborx --parallel 8'
                        dir('build-arborx') {
                            sh 'ctest $CTEST_OPTIONS'
                        }
                        sh 'cmake --install build-arborx'
                        sh 'cmake -S examples -B build-examples -D ArborX_ROOT=$PWD/install-arborx -D Kokkos_ROOT=$PWD/install-kokkos $CMAKE_OPTIONS -DCMAKE_PREFIX_PATH=/opt/rocm'
                        sh 'cmake --build build-examples --parallel 8'
                        dir('build-examples') {
                            sh 'ctest $CTEST_OPTIONS'
                        }
                    }
                    post {
                        always {
                            xunit reduceLog: false, tools:[CTest(deleteOutputFiles: true, failIfNotNew: true, pattern: 'build-*/Testing/**/Test.xml', skipNoTestFiles: false, stopProcessingIfError: true)]
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
