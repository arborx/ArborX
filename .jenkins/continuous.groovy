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
        DOCKER_CREDENTIALS_ID = 'dockerhub'
    }
    stages {

        stage("Style") {
            agent {
                dockerfile {
                    filename "Dockerfile.clang-format"
                    dir "docker"
                    label 'docker'
                    registryCredentialsId "${env.DOCKER_CREDENTIALS_ID}"
                }
            }
            steps {
                sh './scripts/check_format_cpp.sh'
                sh './scripts/check_copyright.sh'
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
