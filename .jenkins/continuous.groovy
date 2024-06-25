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
        CCACHE_MAXSIZE = '5G'
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
                // Disable deprecation warnings since we are using Kokkos::bit_cast which aliases a deprecated function in the oneAPI API.
                stage('SYCL') {
                    agent {
                        dockerfile {
                            filename "Dockerfile.sycl"
                            dir "docker"
                            args '-v /tmp/ccache.kokkos:/tmp/ccache'
                            label 'nvidia-docker && ampere'
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
                                    -D CMAKE_CXX_COMPILER=${DPCPP} \
                                    -D CMAKE_CXX_EXTENSIONS=OFF \
                                    -D CMAKE_CXX_FLAGS="-fp-model=precise -fsycl-device-code-split=per_kernel -Wpedantic -Wall -Wextra -Wno-sycl-target -Wno-unknown-cuda-version -Wno-deprecated-declarations" \
                                    -D CMAKE_PREFIX_PATH="$KOKKOS_DIR;$BOOST_DIR;$BENCHMARK_DIR" \
                                    -D ARBORX_ENABLE_MPI=ON \
                                    -D MPIEXEC_PREFLAGS="--allow-run-as-root" \
                                    -D MPIEXEC_MAX_NUMPROCS=4 \
                                    -D ARBORX_ENABLE_TESTS=ON \
                                    -D ARBORX_ENABLE_EXAMPLES=ON \
                                    -D ARBORX_ENABLE_BENCHMARKS=ON \
                                    -D ARBORX_ENABLE_ONEDPL=ON \
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
                                        -D CMAKE_CXX_COMPILER=${DPCPP} \
                                        -D CMAKE_CXX_EXTENSIONS=OFF \
                                        -D CMAKE_CXX_FLAGS="-Wno-sycl-target -Wno-unknown-cuda-version -Wno-deprecated-declarations" \
                                        -D CMAKE_PREFIX_PATH="$KOKKOS_DIR;$ARBORX_DIR" \
                                        -D MPIEXEC_PREFLAGS="--allow-run-as-root" \
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
