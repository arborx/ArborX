Consider setting the `COMPOSE_PROJECT_NAME` environment variable or providing it
in the environment file.  Its value will be prepended along with the service
name to the container on start up.


You may use multiple compose files to customize your container.  For instance, you
could reproduce the configuration from one of the automated builds by providing
the following a `docker-compose.override.yml` file:
```
version: '3'
services:
  arborx_dev:
    build:
      args:
        - BASE=nvidia/cuda:11.5.2-devel-ubuntu20.04
        - KOKKOS_VERSION=4.0.00
        - KOKKOS_OPTIONS=-DCMAKE_CXX_STANDARD=17 -DKokkos_ENABLE_OPENMP=ON -DKokkos_ENABLE_CUDA=ON -DKokkos_ENABLE_CUDA_LAMBDA=ON -DKokkos_ARCH_SNB=ON -DKokkos_ARCH_VOLTA70=ON
```
