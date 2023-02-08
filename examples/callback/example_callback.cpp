/****************************************************************************
 * Copyright (c) 2017-2022 by the ArborX authors                            *
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

#include <iostream>
#include <random>
#include <vector>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;

struct FirstOctant
{};

struct NearestToOrigin
{
  int k;
};

template <>
struct ArborX::AccessTraits<FirstOctant, ArborX::PredicatesTag>
{
  static KOKKOS_FUNCTION std::size_t size(FirstOctant) { return 1; }
  static KOKKOS_FUNCTION auto get(FirstOctant, std::size_t)
  {
    return ArborX::intersects(ArborX::Box{{{0, 0, 0}}, {{1, 1, 1}}});
  }
  using memory_space = MemorySpace;
};

template <>
struct ArborX::AccessTraits<NearestToOrigin, ArborX::PredicatesTag>
{
  static KOKKOS_FUNCTION std::size_t size(NearestToOrigin) { return 1; }
  static KOKKOS_FUNCTION auto get(NearestToOrigin d, std::size_t)
  {
    return ArborX::nearest(ArborX::Point{0, 0, 0}, d.k);
  }
  using memory_space = MemorySpace;
};

struct PrintfCallback
{
  template <typename Predicate, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Predicate, int primitive,
                                  OutputFunctor const &out) const
  {
#ifdef __SYCL_DEVICE_ONLY__
    using sycl::ext::oneapi::experimental::printf;
#endif
    printf("Found %d from functor\n", primitive);
    out(primitive);
  }
};

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  int const n = 100;
  std::vector<ArborX::Point> points;
  // Fill vector with random points in [-1, 1]^3
  std::uniform_real_distribution<float> dis{-1., 1.};
  std::default_random_engine gen;
  auto rd = [&]() { return dis(gen); };
  std::generate_n(std::back_inserter(points), n, [&]() {
    return ArborX::Point{rd(), rd(), rd()};
  });

  ArborX::BVH<MemorySpace> bvh{
      ExecutionSpace{},
      Kokkos::create_mirror_view_and_copy(
          MemorySpace{},
          Kokkos::View<ArborX::Point *, Kokkos::HostSpace,
                       Kokkos::MemoryUnmanaged>(points.data(), points.size()))};

  {
    Kokkos::View<int *, MemorySpace> values("values", 0);
    Kokkos::View<int *, MemorySpace> offsets("offsets", 0);
    ArborX::query(bvh, ExecutionSpace{}, FirstOctant{}, PrintfCallback{},
                  values, offsets);
#ifndef __NVCC__
    ArborX::query(
        bvh, ExecutionSpace{}, FirstOctant{},
        KOKKOS_LAMBDA(auto /*predicate*/, int primitive,
                      auto /*output_functor*/) {
#ifdef __SYCL_DEVICE_ONLY__
          using sycl::ext::oneapi::experimental::printf;
#endif
          printf("Found %d from generic lambda\n", primitive);
        },
        values, offsets);
#endif
  }

  {
    int const k = 10;
    Kokkos::View<int *, MemorySpace> values("values", 0);
    Kokkos::View<int *, MemorySpace> offsets("offsets", 0);
    ArborX::query(bvh, ExecutionSpace{}, NearestToOrigin{k}, PrintfCallback{},
                  values, offsets);
#ifndef __NVCC__
    ArborX::query(
        bvh, ExecutionSpace{}, NearestToOrigin{k},
        KOKKOS_LAMBDA(auto /*predicate*/, int primitive,
                      auto /*output_functor*/) {
#ifdef __SYCL_DEVICE_ONLY__
          using sycl::ext::oneapi::experimental::printf;
#endif
          printf("Found %d from generic lambda\n", primitive);
        },
        values, offsets);
#endif
  }

  {
    // EXPERIMENTAL
    Kokkos::View<int, ExecutionSpace, Kokkos::MemoryTraits<Kokkos::Atomic>> c(
        "counter");

#ifndef __NVCC__
    bvh.query(
        ExecutionSpace{}, FirstOctant{},
        KOKKOS_LAMBDA(auto /*predicate*/, int j) {
#ifdef __SYCL_DEVICE_ONLY__
          using sycl::ext::oneapi::experimental::printf;
#endif
          printf("%d %d %d\n", ++c(), -1, j);
        });
#endif
  }

  return 0;
}
