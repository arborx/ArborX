/****************************************************************************
 * Copyright (c) 2025, ArborX authors                                       *
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
#include <misc/ArborX_Exception.hpp>
#include <misc/ArborX_SortUtils.hpp>
#include <misc/ArborX_Utils.hpp>

#include <Kokkos_Core.hpp>

#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <numeric>
#include <vector>

#define BOOST_TEST_MODULE Utils

namespace tt = boost::test_tools;

BOOST_AUTO_TEST_CASE_TEMPLATE(sort_objects, DeviceType, ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  ExecutionSpace space{};

  for (auto const &values : {std::vector<int>{36, 19, 25, 17, 3, 7, 1, 2, 9},
                             std::vector<int>{36, 19, 25, 17, 3, 9, 1, 2, 7},
                             std::vector<int>{100, 19, 36, 17, 3, 25, 1, 2, 7},
                             std::vector<int>{15, 5, 11, 3, 4, 8}})
  {
    Kokkos::View<int *, Kokkos::HostSpace> host_view("data", values.size());
    std::copy(values.begin(), values.end(), host_view.data());
    auto device_view = Kokkos::create_mirror_view_and_copy(space, host_view);
    auto device_permutation = ArborX::Details::sortObjects(space, device_view);
    Kokkos::deep_copy(space, host_view, device_view);

    // Check that values were sorted properly
    std::vector<int> values_copy = values;
    std::sort(values_copy.begin(), values_copy.end());
    auto host_permutation = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, device_permutation);
    BOOST_TEST(host_view == values_copy, tt::per_element());

    // Check correctness of the permutation
    for (unsigned int i = 0; i < values.size(); ++i)
      values_copy[i] = values[host_permutation(i)];
    BOOST_TEST(host_view == values_copy, tt::per_element());
  }
}

namespace Test
{
using ArborXTest::toView;

template <class ExecutionSpace>
auto build_offsets(ExecutionSpace const &exec_space,
                   std::vector<int> const &sorted_indices_host)
{
  auto sorted_indices =
      toView<ExecutionSpace>(sorted_indices_host, "Test::sorted_indices");
  Kokkos::View<int *, typename decltype(sorted_indices)::memory_space> offsets(
      "Test::offsets", 0);
  ArborX::Details::computeOffsetsInOrderedView(exec_space, sorted_indices,
                                               offsets);
  return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offsets);
}
} // namespace Test

#define ARBORX_TEST_OFFSETS_IN_SORTED_VIEW(exec_space, sorted_indices, ref)    \
  BOOST_TEST(Test::build_offsets(exec_space, sorted_indices) == ref,           \
             boost::test_tools::per_element());

BOOST_AUTO_TEST_CASE_TEMPLATE(compute_offsets_in_sorted_view, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  ExecutionSpace space{};

  ARBORX_TEST_OFFSETS_IN_SORTED_VIEW(space, (std::vector<int>{}),
                                     (std::vector<int>{0}));
  ARBORX_TEST_OFFSETS_IN_SORTED_VIEW(space, (std::vector<int>{0}),
                                     (std::vector<int>{0, 1}));
  ARBORX_TEST_OFFSETS_IN_SORTED_VIEW(space, (std::vector<int>{0, 0, 1}),
                                     (std::vector<int>{0, 2, 3}));
  ARBORX_TEST_OFFSETS_IN_SORTED_VIEW(space,
                                     (std::vector<int>{0, 1, 6, 6, 6, 6, 11}),
                                     (std::vector<int>{0, 1, 2, 6, 7}));
  ARBORX_TEST_OFFSETS_IN_SORTED_VIEW(space,
                                     (std::vector<int>{14, 5, 5, 5, 3, 3}),
                                     (std::vector<int>{0, 1, 4, 6}));
}
