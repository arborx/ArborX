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
                stage('MemorySanitizer') {
                    agent {
                        docker {
                            image 'ubuntu:23.04'
                            label 'docker'
                        }
                    }
                    environment {
                        CTEST_OPTIONS = '--timeout 180 --no-compress-output -T Test'
                        CMAKE_OPTIONS = '-D CMAKE_BUILD_TYPE=RelWithDebInfo  -D CMAKE_CXX_EXTENSIONS=OFF -D CMAKE_CXX_COMPILER=clang++ -D CMAKE_CXX_FLAGS="-fsanitize=memory"'
                    }
                    steps {
                        sh 'apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y git cmake libboost-program-options-dev libboost-test-dev libbenchmark-dev clang'
                        sh 'rm -rf source* build* install*'
                        sh 'git clone https://github.com/kokkos/kokkos.git --branch develop --depth 1 source-kokkos'
                        dir('source-kokkos') {
                            sh 'git rev-parse --short HEAD'
                        }
                        sh 'cmake -S source-kokkos -B build-kokkos -D CMAKE_INSTALL_PREFIX=$PWD/install-kokkos $CMAKE_OPTIONS'
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
