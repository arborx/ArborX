pipeline {
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
                docker {
                    // arbitrary image that has clang-format version 7.0
                    image "dalg24/arborx_base:19.04.0-cuda-9.2"
                    label 'docker'
                }
            }
            steps {
                sh './scripts/check_format_cpp.sh'
            }
        }

        stage('Build') {
            parallel {
                stage('NVHPC-21.9') {
                    agent {
                        dockerfile {
                            dockerfile 'Dockerfile.nvhpc'
                            dir 'docker'
                            label 'NVIDIA_Tesla_V100-PCIE-32GB && nvidia-docker && large_images'
                            args '--env NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES'
                        }
                    }
                    environment {
                        CTEST_OPTIONS = '--timeout 180 --no-compress-output -T Test'
                        CMAKE_OPTIONS = '-D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_STANDARD=17 -D CMAKE_CXX_EXTENSIONS=OFF'
                    }
                    steps {
                        sh 'cmake -B build -D CMAKE_INSTALL_PREFIX=$PWD/install -D Kokkos_ROOT=$KOKKOS_DIR -D ARBORX_ENABLE_BENCHMARKS=ON -D ARBORX_ENABLE_EXAMPLES=ON -D ARBORX_ENABLE_TESTS=ON $CMAKE_OPTIONS'
                        sh 'cmake --build build --parallel 8'
                        dir('build') {
                            sh 'ctest $CTEST_OPTIONS'
                        }
                        sh 'cmake --install build'
                        sh 'cmake -S examples -B build-examples -D ArborX_ROOT=$PWD/install -D Kokkos_ROOT=$KOKKOS_DIR $CMAKE_OPTIONS'
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
