/****************************************************************************
 * Copyright (c) 2017-2023 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include "ArborXTest_StdVectorToKokkosView.hpp"
#include "ArborX_EnableDeviceTypes.hpp" // ARBORX_DEVICE_TYPES
#include "ArborX_EnableViewComparison.hpp"
#include <ArborX_DetailsHalfTraversal.hpp>
#include <ArborX_LinearBVH.hpp>

#include "BoostTest_CUDA_clang_workarounds.hpp"
#include <boost/test/unit_test.hpp>

namespace Test
{
template <class ExecutionSpace>
Kokkos::View<ArborX::Point *, ExecutionSpace>
make_points(ExecutionSpace const &space, int n)
{
  Kokkos::View<ArborX::Point *, ExecutionSpace> points(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing, "Test::points"),
      n);

  Kokkos::parallel_for(
      "Test::make_points", Kokkos::RangePolicy<ExecutionSpace>(space, 0, n),
      KOKKOS_LAMBDA(int i) {
        points(i) = {(float)i, (float)i, (float)i};
      });

  return points;
}

// Workaround NVCC's extended lambda restrictions
struct AlwaysTrue
{
  template <class ValueType>
  KOKKOS_FUNCTION bool operator()(ValueType const &) const
  {
    return true;
  }
};

struct PredicateGetter
{
  template <class ValueType>
  KOKKOS_FUNCTION AlwaysTrue operator()(ValueType) const
  {
    return {};
  }
};
} // namespace Test

BOOST_AUTO_TEST_CASE_TEMPLATE(half_traversal, DeviceType, ARBORX_DEVICE_TYPES)
{
  using MemorySpace = typename DeviceType::memory_space;
  using ExecutionSpace = typename DeviceType::execution_space;
  ExecutionSpace exec_space;
  int const n = 24;
  auto points = Test::make_points(exec_space, n);
  ArborX::BVH<MemorySpace> bvh(exec_space, points);

  Kokkos::View<int *, MemorySpace> count("Test::count", n * (n + 1) / 2);

  //    [0][1][2]...[j][i].........[n]
  // [0] 0
  // [1] 1  0
  // [2] 1  1  0
  // ... 1  1  1  0       i*(i+1)/2+i
  // ... 1  1  1  1  0   /
  // [i] 1  1  1  1  1  0
  // ... 1  1  1    /
  // ... 1  1  1   i*(i+1)/2+j
  // ... 1  1  1
  // [n] 1  1  1  1  1  1  1  1  1  0

  using ArborX::Details::HalfTraversal;
  HalfTraversal(
      exec_space, bvh,
      KOKKOS_LAMBDA(int i, int j) {
        auto [min_ij, max_ij] = Kokkos::minmax(i, j);
        Kokkos::atomic_increment(&count(max_ij * (max_ij + 1) / 2 + min_ij));
      },
      Test::PredicateGetter{});

  std::vector<int> count_ref(n * (n + 1) / 2, 1);
  for (int i = 0; i < n; ++i)
  {
    count_ref[i * (i + 1) / 2 + i] = 0;
  }

  BOOST_TEST(count_ref == Kokkos::create_mirror_view_and_copy(
                              Kokkos::HostSpace{}, count),
             boost::test_tools::per_element());
}
