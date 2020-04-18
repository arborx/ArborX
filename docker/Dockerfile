ARG BASE=nvidia/cuda:10.1-devel
FROM $BASE

ARG NPROCS=4

RUN apt-get update && apt-get install -y \
        build-essential \
        bc \
        curl \
        git \
        wget \
        jq \
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

# Disable IPv6 (https://rvm.io/rvm/security#ipv6-issues)
RUN mkdir -p ~/.gnupg && echo "disable-ipv6" >> ~/.gnupg/dirmngr.conf

# Install CMake
ENV CMAKE_DIR=/opt/cmake
RUN CMAKE_VERSION=3.13.4 && \
    CMAKE_KEY=2D2CEF1034921684 && \
    CMAKE_URL=https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION} && \
    CMAKE_SCRIPT=cmake-${CMAKE_VERSION}-Linux-x86_64.sh && \
    CMAKE_SHA256=cmake-${CMAKE_VERSION}-SHA-256.txt && \
    wget --quiet ${CMAKE_URL}/${CMAKE_SHA256} && \
    wget --quiet ${CMAKE_URL}/${CMAKE_SHA256}.asc && \
    wget --quiet ${CMAKE_URL}/${CMAKE_SCRIPT} && \
    gpg --recv-keys ${CMAKE_KEY} && \
    gpg --verify ${CMAKE_SHA256}.asc ${CMAKE_SHA256} && \
    grep ${CMAKE_SCRIPT} ${CMAKE_SHA256} | sha256sum --check && \
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
    gpg --recv-keys ${LLVM_KEY} && \
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
    ../openmpi/configure --prefix=${OPENMPI_DIR} ${CUDA_OPTIONS} CFLAGS=-w && \
    make -j${NPROCS} install && \
    rm -rf ${SCRATCH_DIR}
ENV PATH=${OPENMPI_DIR}/bin:$PATH

# Install Boost
ENV BOOST_DIR=/opt/boost
RUN BOOST_VERSION=1.67.0 && \
    BOOST_VERSION_UNDERSCORE=$(echo "$BOOST_VERSION" | sed -e "s/\./_/g") && \
    BOOST_KEY=379CE192D401AB61 && \
    BOOST_URL=https://dl.bintray.com/boostorg/release/${BOOST_VERSION}/source && \
    BOOST_ARCHIVE=boost_${BOOST_VERSION_UNDERSCORE}.tar.bz2 && \
    wget --quiet ${BOOST_URL}/${BOOST_ARCHIVE} && \
    wget --quiet ${BOOST_URL}/${BOOST_ARCHIVE}.asc && \
    wget --quiet ${BOOST_URL}/${BOOST_ARCHIVE}.json && \
    wget --quiet ${BOOST_URL}/${BOOST_ARCHIVE}.json.asc && \
    gpg --recv-keys ${BOOST_KEY} && \
    gpg --verify ${BOOST_ARCHIVE}.json.asc ${BOOST_ARCHIVE}.json && \
    gpg --verify ${BOOST_ARCHIVE}.asc ${BOOST_ARCHIVE} && \
    cat ${BOOST_ARCHIVE}.json | jq -r '. | .sha256 + "  " + .file' | sha256sum --check && \
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
        cxxflags=-w \
        install \
        && \
    rm -rf boost*

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

# Workaround for Kokkos to find libcudart
ENV LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}

# Install Kokkos
ARG KOKKOS_VERSION=3.1.00
ARG KOKKOS_OPTIONS="-DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON"
ENV KOKKOS_DIR=/opt/kokkos
RUN KOKKOS_URL=https://github.com/kokkos/kokkos/archive/${KOKKOS_VERSION}.tar.gz && \
    KOKKOS_ARCHIVE=kokkos-${KOKKOS_HASH}.tar.gz && \
    SCRATCH_DIR=/scratch && mkdir -p ${SCRATCH_DIR} && cd ${SCRATCH_DIR} && \
    wget --quiet ${KOKKOS_URL} --output-document=${KOKKOS_ARCHIVE} && \
    mkdir -p kokkos && \
    tar -xf ${KOKKOS_ARCHIVE} -C kokkos --strip-components=1 && \
    cd kokkos && \
    mkdir -p build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=${KOKKOS_DIR} -D CMAKE_CXX_COMPILER=/scratch/kokkos/bin/nvcc_wrapper ${KOKKOS_OPTIONS} .. && \
    make -j${NPROCS} install && \
    rm -rf ${SCRATCH_DIR}
