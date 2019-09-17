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
        - BASE=nvidia/cuda:10.1-devel
        - KOKKOS_VERSION=2.9.00
        - KOKKOS_OPTIONS=--cxxstandard=c++14 --with-serial --with-openmp --with-options=disable_deprecated_code --with-cuda --with-cuda-options=enable_lambda --arch=SNB,Volta70
```
