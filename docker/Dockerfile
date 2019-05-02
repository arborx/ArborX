ARG BASE=ubuntu:18.04
FROM $BASE

ARG NPROCS=4

RUN apt-get update && apt-get install -y \
        build-essential \
        bc \
        curl \
        git \
        wget \
        vim \
        lcov \
        ccache \
        gdb \
        ninja-build \
        libbz2-dev \
        libicu-dev \
        python-dev \
        autotools-dev \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install CMake
ENV CMAKE_DIR=/opt/cmake
RUN CMAKE_VERSION=3.13.4 && \
    CMAKE_VERSION_SHORT=3.13 && \
    CMAKE_URL=https://cmake.org/files/v${CMAKE_VERSION_SHORT}/cmake-${CMAKE_VERSION}-Linux-x86_64.sh && \
    CMAKE_SCRIPT=cmake-${CMAKE_VERSION}-Linux-x86_64.sh && \
    wget --quiet ${CMAKE_URL} --output-document=${CMAKE_SCRIPT} && \
    mkdir -p ${CMAKE_DIR} && \
    sh ${CMAKE_SCRIPT} --skip-license --prefix=${CMAKE_DIR} && \
    rm ${CMAKE_SCRIPT}
ENV PATH=${CMAKE_DIR}/bin:$PATH

# Install Clang/LLVM
ENV LLVM_DIR=/opt/llvm
RUN LLVM_VERSION=7.0.1 && \
    LLVM_KEY=86419D8A && \
    LLVM_URL=http://releases.llvm.org/${LLVM_VERSION}/clang+llvm-${LLVM_VERSION}-x86_64-linux-gnu-ubuntu-18.04.tar.xz && \
    LLVM_ARCHIVE=llvm-${LLVM_VERSION}.tar.xz && \
    SCRATCH_DIR=/scratch && mkdir -p ${SCRATCH_DIR} && cd ${SCRATCH_DIR} && \
    wget --quiet ${LLVM_URL} --output-document=${LLVM_ARCHIVE} && \
    wget --quiet ${LLVM_URL}.sig --output-document=${LLVM_ARCHIVE}.sig && \
    gpg --keyserver pool.sks-keyservers.net --recv-keys ${LLVM_KEY} && \
    gpg --verify ${LLVM_ARCHIVE}.sig ${LLVM_ARCHIVE} && \
    mkdir -p ${LLVM_DIR} && \
    tar -xvf ${LLVM_ARCHIVE} -C ${LLVM_DIR} --strip-components=1 && \
    echo "${LLVM_DIR}/lib" > /etc/ld.so.conf.d/llvm.conf && ldconfig && \
    rm -rf /root/.gnupg && \
    rm -rf ${SCRATCH_DIR}
ENV PATH=${LLVM_DIR}/bin:$PATH

# Install OpenMPI
ENV OPENMPI_DIR=/opt/openmpi
RUN OPENMPI_VERSION=3.1.3 && \
    OPENMPI_VERSION_SHORT=3.1 && \
    OPENMPI_SHA1=b3c60e2bdd5a8a8e758fd741f9a5bebb84da5e81 && \
    OPENMPI_URL=https://download.open-mpi.org/release/open-mpi/v${OPENMPI_VERSION_SHORT}/openmpi-${OPENMPI_VERSION}.tar.bz2 && \
    OPENMPI_ARCHIVE=openmpi-${OPENMPI_VERSION}.tar.bz2 && \
    [ ! -z "${CUDA_VERSION}" ] && CUDA_OPTIONS=--with-cuda || true && \
    SCRATCH_DIR=/scratch && mkdir -p ${SCRATCH_DIR} && cd ${SCRATCH_DIR} && \
    wget --quiet ${OPENMPI_URL} --output-document=${OPENMPI_ARCHIVE} && \
    echo "${OPENMPI_SHA1} ${OPENMPI_ARCHIVE}" | sha1sum -c && \
    mkdir -p openmpi && \
    tar -xf ${OPENMPI_ARCHIVE} -C openmpi --strip-components=1 && \
    mkdir -p build && cd build && \
    ../openmpi/configure --prefix=${OPENMPI_DIR} ${CUDA_OPTIONS} && \
    make -j${NPROCS} install && \
    rm -rf ${SCRATCH_DIR}
ENV PATH=${OPENMPI_DIR}/bin:$PATH

# Install Boost
ENV BOOST_DIR=/opt/boost
RUN BOOST_VERSION=1.67.0 && \
    BOOST_VERSION_UNDERSCORE=$(echo "$BOOST_VERSION" | sed -e "s/\./_/g") && \
    BOOST_URL=https://dl.bintray.com/boostorg/release/${BOOST_VERSION}/source/boost_${BOOST_VERSION_UNDERSCORE}.tar.bz2 && \
    BOOST_SHA256=2684c972994ee57fc5632e03bf044746f6eb45d4920c343937a465fd67a5adba && \
    BOOST_ARCHIVE=boost_${BOOST_VERSION_UNDERSCORE}.tar.bz2 && \
    SCRATCH_DIR=/scratch && mkdir -p ${SCRATCH_DIR} && cd ${SCRATCH_DIR} && \
    wget --quiet ${BOOST_URL} --output-document=${BOOST_ARCHIVE} && \
    echo "${BOOST_SHA256} ${BOOST_ARCHIVE}" | sha256sum -c && \
    mkdir -p boost && \
    tar -xf ${BOOST_ARCHIVE} -C boost --strip-components=1 && \
    cd boost && \
    ./bootstrap.sh \
        --prefix=${BOOST_DIR} \
        && \
    echo "using mpi ;" >> project-config.jam && \
    ./b2 -j${NPROCS} \
        hardcode-dll-paths=true dll-path=${BOOST_DIR}/lib \
        link=shared \
        variant=release \
        install \
        && \
    rm -rf ${SCRATCH_DIR}

# Install Google Benchmark support library
ENV BENCHMARK_DIR=/opt/benchmark
RUN SCRATCH_DIR=/scratch && mkdir -p ${SCRATCH_DIR} && cd ${SCRATCH_DIR} && \
    git clone https://github.com/google/benchmark.git -b v1.4.1 && \
    cd benchmark && \
    git clone https://github.com/google/googletest.git -b release-1.8.1 && \
    mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=${BENCHMARK_DIR} .. && \
    make -j${NPROCS} && make install && \
    rm -rf ${SCRATCH_DIR}

# Install Kokkos
ENV KOKKOS_DIR=/opt/kokkos
RUN KOKKOS_VERSION=5d6e7fb38e96aec88d2c514e1f9be1cf2b549b57 && \
    KOKKOS_URL=https://github.com/kokkos/kokkos/archive/${KOKKOS_VERSION}.tar.gz && \
    KOKKOS_ARCHIVE=kokkos-${KOKKOS_HASH}.tar.gz && \
    SCRATCH_DIR=/scratch && mkdir -p ${SCRATCH_DIR} && cd ${SCRATCH_DIR} && \
    wget --quiet ${KOKKOS_URL} --output-document=${KOKKOS_ARCHIVE} && \
    mkdir -p kokkos && \
    tar -xf ${KOKKOS_ARCHIVE} -C kokkos --strip-components=1 && \
    cd kokkos && \
    mkdir -p build && cd build && \
    [ ! -z "${CUDA_VERSION}" ] && CUDA_OPTIONS="--with-cuda --with-cuda-options=enable_lambda" || true && \
    ../generate_makefile.bash --prefix=${KOKKOS_DIR} \
       --with-serial --with-openmp \
       --with-options=disable_deprecated_code ${CUDA_OPTIONS} \
       --arch=SNB,Volta70 \
       --cxxstandard=c++14 && \
    make -j${NPROCS} install && \
    rm -rf ${SCRATCH_DIR}
