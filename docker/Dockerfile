ARG BASE=ubuntu:22.04
FROM $BASE

ARG NPROCS=4

ARG ADDITIONAL_PACKAGES=""
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -yq \
        ${ADDITIONAL_PACKAGES} \
        build-essential \
        ccache \
        cmake \
        curl \
        git \
        libomp-dev \
        libopenmpi-dev \
        libbenchmark-dev \
        libboost-program-options-dev \
        libboost-test-dev \
        vim \
        wget \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN KEYDUMP_URL=https://cloud1.cees.ornl.gov/download && \
    KEYDUMP_FILE=keydump && \
    wget --quiet ${KEYDUMP_URL}/${KEYDUMP_FILE} && \
    wget --quiet ${KEYDUMP_URL}/${KEYDUMP_FILE}.sig && \
    gpg --import ${KEYDUMP_FILE} && \
    gpg --verify ${KEYDUMP_FILE}.sig ${KEYDUMP_FILE} && \
    rm ${KEYDUMP_FILE}*

# Install Kokkos
ARG KOKKOS_VERSION=4.5.00
ARG KOKKOS_OPTIONS="-DCMAKE_CXX_COMPILER=clang++ -DCMAKE_CXX_STANDARD=20 -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=ON"
ENV KOKKOS_DIR=/opt/kokkos
RUN KOKKOS_URL=https://github.com/kokkos/kokkos/archive/${KOKKOS_VERSION}.tar.gz && \
    KOKKOS_ARCHIVE=kokkos-${KOKKOS_VERSION}.tar.gz && \
    SCRATCH_DIR=/scratch && mkdir -p ${SCRATCH_DIR} && cd ${SCRATCH_DIR} && \
    wget --quiet ${KOKKOS_URL} --output-document=${KOKKOS_ARCHIVE} && \
    mkdir -p kokkos && \
    tar -xf ${KOKKOS_ARCHIVE} -C kokkos --strip-components=1 && \
    cd kokkos && \
    mkdir -p build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=${KOKKOS_DIR} ${KOKKOS_OPTIONS} .. && \
    make -j${NPROCS} install && \
    rm -rf ${SCRATCH_DIR}
