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

// clang-format off
#include "boost_ext/KokkosPairComparison.hpp"
#include "boost_ext/TupleComparison.hpp"
#include "boost_ext/CompressedStorageComparison.hpp"
// clang-format on

#include "ArborXTest_Cloud.hpp"
#include "ArborXTest_StdVectorToKokkosView.hpp"
#include "ArborX_EnableDeviceTypes.hpp" // ARBORX_DEVICE_TYPES
#include "ArborX_EnableViewComparison.hpp"
#include <ArborX_Sphere.hpp>
#include <detail/ArborX_ExpandHalfToFull.hpp>
#include <detail/ArborX_NeighborList.hpp>

#include <Kokkos_Random.hpp>

#include "BoostTest_CUDA_clang_workarounds.hpp"
#include <boost/test/unit_test.hpp>

namespace Test
{
using ArborXTest::toView;

struct Filter
{
  template <class Predicate, typename Value, class OutputFunctor>
  KOKKOS_FUNCTION void operator()(Predicate const &predicate,
                                  Value const &value,
                                  OutputFunctor const &out) const
  {
    int const i = value.index;
    int const j = getData(predicate);
    if (i < j)
    {
      out(i);
    }
  }
};

template <class MemorySpace, class ExecutionSpace, class Points>
auto compute_reference(ExecutionSpace const &exec_space, Points const &points,
                       float radius)
{
  Kokkos::View<int *, ExecutionSpace> offsets("Test::offsets", 0);
  Kokkos::View<int *, ExecutionSpace> indices("Test::indices", 0);
  ArborX::BoundingVolumeHierarchy bvh(
      exec_space, ArborX::Experimental::attach_indices(points));
  auto predicates = ArborX::Experimental::attach_indices<int>(
      ArborX::Experimental::make_intersects(points, radius));
  bvh.query(exec_space, predicates, Filter{}, indices, offsets);
  ArborX::Details::expandHalfToFull(exec_space, offsets, indices);
  return make_compressed_storage(
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offsets),
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, indices));
}

template <class ExecutionSpace, class Points>
auto buildFullNeighborList(ExecutionSpace const &exec_space,
                           Points const &points, float radius)
{
  Kokkos::View<int *, ExecutionSpace> offsets("Test::offsets", 0);
  Kokkos::View<int *, ExecutionSpace> indices("Test::indices", 0);
  ArborX::Experimental::findFullNeighborList(exec_space, points, radius,
                                             offsets, indices);
  return make_compressed_storage(
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offsets),
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, indices));
}

template <class ExecutionSpace, class Points>
auto buildHalfNeighborListAndExpandToFull(ExecutionSpace const &exec_space,
                                          Points const &points, float radius)
{
  Kokkos::View<int *, ExecutionSpace> offsets("Test::offsets", 0);
  Kokkos::View<int *, ExecutionSpace> indices("Test::indices", 0);
  ArborX::Experimental::findHalfNeighborList(exec_space, points, radius,
                                             offsets, indices);
  ArborX::Details::expandHalfToFull(exec_space, offsets, indices);
  return make_compressed_storage(
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offsets),
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, indices));
}

#define ARBORX_TEST_NEIGHBOR_LIST(exec_space, points, radius, offsets_ref,     \
                                  indices_ref)                                 \
  BOOST_TEST(Test::buildFullNeighborList(exec_space, points, radius) ==        \
                 make_compressed_storage(offsets_ref, indices_ref),            \
             boost::test_tools::per_element());                                \
  BOOST_TEST(Test::buildHalfNeighborListAndExpandToFull(exec_space, points,    \
                                                        radius) ==             \
                 make_compressed_storage(offsets_ref, indices_ref),            \
             boost::test_tools::per_element())

} // namespace Test

BOOST_AUTO_TEST_CASE_TEMPLATE(find_neighbor_list_degenerate, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  ExecutionSpace exec_space;

  auto no_point = ArborXTest::toView<ExecutionSpace>(
      std::vector<ArborX::Point<3>>{}, "Test::no_point");

  auto single_point = ArborXTest::toView<ExecutionSpace>(
      std::vector<ArborX::Point<3>>{{0.f, 0.f, 0.f}}, "Test::single_point");

  constexpr auto radius =
      ArborX::Details::KokkosExt::ArithmeticTraits::infinity<float>::value;

  ARBORX_TEST_NEIGHBOR_LIST(exec_space, no_point, radius, (std::vector<int>{0}),
                            (std::vector<int>{}));

  ARBORX_TEST_NEIGHBOR_LIST(exec_space, single_point, radius,
                            (std::vector<int>{0, 0}), (std::vector<int>{}));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(find_neighbor_list, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  ExecutionSpace exec_space;

  auto points = ArborXTest::toView<ExecutionSpace>(
      std::vector<ArborX::Point<3>>{
          {0.f, 0.f, 0.f},
          {1.f, 1.f, 1.f},
          {2.f, 2.f, 2.f},
          {3.f, 3.f, 3.f},
      },
      "Test::four_points");

  ARBORX_TEST_NEIGHBOR_LIST(exec_space, points, 1.f,
                            (std::vector<int>{0, 0, 0, 0, 0}),
                            (std::vector<int>{}));

  ARBORX_TEST_NEIGHBOR_LIST(exec_space, points, 2.f,
                            (std::vector<int>{0, 1, 3, 5, 6}),
                            (std::vector<int>{1, 0, 2, 1, 3, 2}));

  ARBORX_TEST_NEIGHBOR_LIST(exec_space, points, 4.f,
                            (std::vector<int>{0, 2, 5, 8, 10}),
                            (std::vector<int>{1, 2, 0, 2, 3, 0, 1, 3, 1, 2}));

  ARBORX_TEST_NEIGHBOR_LIST(
      exec_space, points, 6.f, (std::vector<int>{0, 3, 6, 9, 12}),
      (std::vector<int>{1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2}));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(
    find_neighbor_list_compare_filtered_tree_traversal, DeviceType,
    ARBORX_DEVICE_TYPES)
{
  using MemorySpace = typename DeviceType::memory_space;
  using ExecutionSpace = typename DeviceType::execution_space;
  ExecutionSpace exec_space;

  auto points =
      ArborXTest::make_random_cloud<ArborX::Point<3>>(exec_space, 100);
  auto radius = .3f;

  BOOST_TEST(
      Test::buildFullNeighborList(exec_space, points, radius) ==
          Test::compute_reference<MemorySpace>(exec_space, points, radius),
      boost::test_tools::per_element());
  BOOST_TEST(
      Test::buildHalfNeighborListAndExpandToFull(exec_space, points, radius) ==
          Test::compute_reference<MemorySpace>(exec_space, points, radius),
      boost::test_tools::per_element());
}
