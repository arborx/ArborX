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
#include "boost_ext/CompressedStorageComparison.hpp"
// clang-format on

#include "ArborXTest_StdVectorToKokkosView.hpp"
#include "ArborX_EnableDeviceTypes.hpp" // ARBORX_DEVICE_TYPES
#include "ArborX_EnableViewComparison.hpp"
#include <ArborX_DetailsExpandHalfToFull.hpp>

#include "BoostTest_CUDA_clang_workarounds.hpp"
#include <boost/test/unit_test.hpp>

namespace Test
{
using ArborXTest::toView;

template <class ExecutionSpace>
auto expand(ExecutionSpace space, std::vector<int> const &offsets_host,
            std::vector<int> const &indices_host)
{
  auto offsets = toView<ExecutionSpace>(offsets_host, "Test::offsets");
  auto indices = toView<ExecutionSpace>(indices_host, "Test::indices");
  ArborX::Details::expandHalfToFull(space, offsets, indices);

  return make_compressed_storage(
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offsets),
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, indices));
}

#define ARBORX_TEST_EXPAND_HALF_TO_FULL(exec_space, offsets_in, indices_in,    \
                                        offsets_out, indices_out)              \
  BOOST_TEST(Test::expand(exec_space, offsets_in, indices_in) ==               \
                 make_compressed_storage(offsets_out, indices_out),            \
             boost::test_tools::per_element())

} // namespace Test

BOOST_AUTO_TEST_CASE_TEMPLATE(expand_half_to_full, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  ExecutionSpace exec_space;

  ARBORX_TEST_EXPAND_HALF_TO_FULL(exec_space, (std::vector<int>{0}),
                                  (std::vector<int>{}), (std::vector<int>{0}),
                                  (std::vector<int>{}));

  // (0,0) -> (0,0)(0,0)
  ARBORX_TEST_EXPAND_HALF_TO_FULL(
      exec_space, (std::vector<int>{0, 1}), (std::vector<int>{0}),
      (std::vector<int>{0, 2}), (std::vector<int>{0, 0}));

  // (0,0)(0,0) -> (0,0)(0,0)(0,0)(0,0)
  ARBORX_TEST_EXPAND_HALF_TO_FULL(
      exec_space, (std::vector<int>{0, 2}), (std::vector<int>{0, 0}),
      (std::vector<int>{0, 4}), (std::vector<int>{0, 0, 0, 0}));

  // (0,0)(0,1) -> (0,0)(0,1)(0,0)(1,0)
  ARBORX_TEST_EXPAND_HALF_TO_FULL(
      exec_space, (std::vector<int>{0, 2, 2}), (std::vector<int>{0, 1}),
      (std::vector<int>{0, 3, 4}), (std::vector<int>{0, 1, 0, 0}));

  // (1,0) -> (0,1)(1,0)
  ARBORX_TEST_EXPAND_HALF_TO_FULL(
      exec_space, (std::vector<int>{0, 0, 1, 1, 1}), (std::vector<int>{0}),
      (std::vector<int>{0, 1, 2, 2, 2}), (std::vector<int>{1, 0}));
}
