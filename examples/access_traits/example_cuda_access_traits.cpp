/****************************************************************************
 * Copyright (c) 2012-2020 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <ArborX.hpp>

#include <Kokkos_Core.hpp>

#include <array>
#include <iostream>
#include <numeric>

struct PointCloud
{
  float *d_x;
  float *d_y;
  float *d_z;
  int N;
};

struct Spheres
{
  float *d_x;
  float *d_y;
  float *d_z;
  float *d_r;
  int N;
};

namespace ArborX
{
template <>
struct AccessTraits<PointCloud, PrimitivesTag>
{
  static std::size_t size(PointCloud const &cloud) { return cloud.N; }
  KOKKOS_FUNCTION static Point get(PointCloud const &cloud, std::size_t i)
  {
    return {{cloud.d_x[i], cloud.d_y[i], cloud.d_z[i]}};
  }
  using memory_space = Kokkos::CudaSpace;
};

template <>
struct AccessTraits<Spheres, PredicatesTag>
{
  static std::size_t size(Spheres const &d) { return d.N; }
  KOKKOS_FUNCTION static auto get(Spheres const &d, std::size_t i)
  {
    return intersects(Sphere{{{d.d_x[i], d.d_y[i], d.d_z[i]}}, d.d_r[i]});
  }
  using memory_space = Kokkos::CudaSpace;
};
} // namespace ArborX

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  constexpr std::size_t N = 10;
  std::array<float, N> a;

  float *d_a;
  cudaMalloc(&d_a, sizeof(a));

  std::iota(std::begin(a), std::end(a), 1.0);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cudaMemcpyAsync(d_a, a.data(), sizeof(a), cudaMemcpyHostToDevice, stream);

  Kokkos::Cuda cuda{stream};
  ArborX::BVH<Kokkos::CudaSpace> bvh{cuda, PointCloud{d_a, d_a, d_a, N}};

  Kokkos::View<int *, Kokkos::CudaSpace> indices("indices", 0);
  Kokkos::View<int *, Kokkos::CudaSpace> offset("offset", 0);
  bvh.query(cuda, Spheres{d_a, d_a, d_a, d_a, N}, indices, offset);

  Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::Cuda>(cuda, 0, N),
                       KOKKOS_LAMBDA(int i) {
                         for (int j = offset(i); j < offset(i + 1); ++j)
                         {
                           printf("%i %i\n", i, indices(j));
                         }
                       });

  cudaStreamDestroy(stream);

  return 0;
}
