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
#include "ArborX_EnableDeviceTypes.hpp" // ARBORX_DEVICE_TYPES
#include "ArborX_EnableViewComparison.hpp"
#include <ArborX_DetailsUnionFind.hpp>
#include <ArborX_DetailsUtils.hpp> // iota

#include "BoostTest_CUDA_clang_workarounds.hpp"
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(UnionFind)

template <typename ExecutionSpace, typename UnionFind>
Kokkos::View<int *, Kokkos::HostSpace>
build_representatives(ExecutionSpace const &space, UnionFind union_find)
{
  using MemorySpace = typename UnionFind::memory_space;

  auto const n = union_find.size();
  Kokkos::View<int *, MemorySpace> representatives(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "Test::representatives"),
      n);
  Kokkos::View<int *, MemorySpace> map2smallest(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing, "Test::map"), n);
  Kokkos::deep_copy(space, map2smallest, INT_MAX);

  Kokkos::parallel_for(
      "Test::find_representatives",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n), KOKKOS_LAMBDA(int i) {
        auto r = union_find.representative(i);
#if KOKKOS_VERSION >= 30799
        Kokkos::atomic_min(&map2smallest(r), i);
#else
         // Workaround for undefined
        //   desul::atomic_min(int*, int, desul::MemoryOrderRelaxed,
        //   desul::MemoryScopeDevice)
        // in older Kokkos versions.
        auto v = map2smallest(r);
        while (v > i) {
          v = Kokkos::atomic_compare_exchange(&map2smallest(r), v, i);
        }
#endif
        representatives(i) = r;
      });
  // We want the representative values to not depend on a specific
  // implementation of the union-find (e.g., not relying them being equal to
  // the smallest index in the set), so we explicitly remap them.
  Kokkos::parallel_for(
      "Test::remap_representatives",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n), KOKKOS_LAMBDA(int i) {
        representatives(i) = map2smallest(representatives(i));
      });

  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                             representatives);
}

template <typename ExecutionSpace, typename UnionFind>
void merge(ExecutionSpace const &space, UnionFind &union_find, int i, int j)
{
  Kokkos::parallel_for(
      "Test::merge", Kokkos::RangePolicy<ExecutionSpace>(space, 0, 1),
      KOKKOS_LAMBDA(int) { union_find.merge(i, j); });
}

#define ARBORX_TEST_UNION_FIND_REPRESENTATIVES(space, union_find, ref)         \
  BOOST_TEST(build_representatives(space, union_find) == ref,                  \
             boost::test_tools::per_element());

BOOST_AUTO_TEST_CASE_TEMPLATE(union_find, DeviceType, ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;

#ifdef KOKKOS_ENABLE_SERIAL
  using UnionFind = ArborX::Details::UnionFind<
      MemorySpace,
      /*DoSerial=*/std::is_same_v<ExecutionSpace, Kokkos::Serial>>;
#else
  using UnionFind = ArborX::Details::UnionFind<MemorySpace>;
#endif

  ExecutionSpace space;

  constexpr int n = 5;

  Kokkos::View<int *, MemorySpace> labels(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing, "Test::labels"),
      n);
  ArborX::iota(space, labels);
  UnionFind union_find(labels);

  ARBORX_TEST_UNION_FIND_REPRESENTATIVES(space, union_find,
                                         (std::vector<int>{0, 1, 2, 3, 4}));

  merge(space, union_find, 1, 1);
  ARBORX_TEST_UNION_FIND_REPRESENTATIVES(space, union_find,
                                         (std::vector<int>{0, 1, 2, 3, 4}));

  merge(space, union_find, 3, 0);
  ARBORX_TEST_UNION_FIND_REPRESENTATIVES(space, union_find,
                                         (std::vector<int>{0, 1, 2, 0, 4}));

  merge(space, union_find, 1, 2);
  merge(space, union_find, 4, 1);
  merge(space, union_find, 1, 1);
  ARBORX_TEST_UNION_FIND_REPRESENTATIVES(space, union_find,
                                         (std::vector<int>{0, 1, 1, 0, 1}));

  merge(space, union_find, 0, 1);
  ARBORX_TEST_UNION_FIND_REPRESENTATIVES(space, union_find,
                                         (std::vector<int>{0, 0, 0, 0, 0}));
}

BOOST_AUTO_TEST_SUITE_END()
