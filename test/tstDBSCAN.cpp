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
#include "ArborXTest_StdVectorToKokkosView.hpp"
#include "ArborX_EnableDeviceTypes.hpp" // ARBORX_DEVICE_TYPES
#include <ArborX_DBSCAN.hpp>
#include <ArborX_DBSCANVerification.hpp>

#include "BoostTest_CUDA_clang_workarounds.hpp"
#include <boost/test/unit_test.hpp>

template <typename View>
struct HiddenView
{
  View _view;
};
template <typename View>
struct ArborX::AccessTraits<HiddenView<View>, ArborX::PrimitivesTag>
{
  using Data = HiddenView<View>;
  static KOKKOS_FUNCTION std::size_t size(Data const &data)
  {
    return data._view.extent(0);
  }
  static KOKKOS_FUNCTION typename View::value_type const &get(Data const &data,
                                                              std::size_t i)
  {
    return data._view(i);
  }
  using memory_space = typename View::memory_space;
};

using ArborXTest::toView;

BOOST_AUTO_TEST_SUITE(DBSCAN)

BOOST_AUTO_TEST_CASE_TEMPLATE(dbscan_verifier, DeviceType, ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using ArborX::Point;
  using ArborX::Details::verifyDBSCAN;

  ExecutionSpace space;

  {
    auto points = toView<DeviceType, Point>({{{0, 0, 0}}, {{1, 1, 1}}});

    auto r = std::sqrt(3);

    BOOST_TEST(verifyDBSCAN(space, points, r - 0.1, 2,
                            toView<DeviceType, int>({-1, -1})));
    BOOST_TEST(!verifyDBSCAN(space, points, r - 0.1, 2,
                             toView<DeviceType, int>({1, 2})));
    BOOST_TEST(!verifyDBSCAN(space, points, r - 0.1, 2,
                             toView<DeviceType, int>({1, 1})));
    BOOST_TEST(
        verifyDBSCAN(space, points, r, 2, toView<DeviceType, int>({1, 1})));
    BOOST_TEST(
        !verifyDBSCAN(space, points, r, 2, toView<DeviceType, int>({1, 2})));
    BOOST_TEST(
        verifyDBSCAN(space, points, r, 3, toView<DeviceType, int>({-1, -1})));
    BOOST_TEST(
        !verifyDBSCAN(space, points, r, 3, toView<DeviceType, int>({1, 1})));
  }

  {
    auto points = toView<DeviceType, Point>(
        {{{0, 0, 0}}, {{1, 1, 1}}, {{3, 3, 3}}, {{6, 6, 6}}});

    auto r = std::sqrt(3);

    BOOST_TEST(verifyDBSCAN(space, points, r, 2,
                            toView<DeviceType, int>({1, 1, -1, -1})));
    BOOST_TEST(verifyDBSCAN(space, points, r, 3,
                            toView<DeviceType, int>({-1, -1, -1, -1})));

    BOOST_TEST(verifyDBSCAN(space, points, 2 * r, 2,
                            toView<DeviceType, int>({3, 3, 3, -1})));
    BOOST_TEST(verifyDBSCAN(space, points, 2 * r, 3,
                            toView<DeviceType, int>({3, 3, 3, -1})));
    BOOST_TEST(verifyDBSCAN(space, points, 2 * r, 4,
                            toView<DeviceType, int>({-1, -1, -1, -1})));

    BOOST_TEST(verifyDBSCAN(space, points, 3 * r, 2,
                            toView<DeviceType, int>({5, 5, 5, 5})));
    BOOST_TEST(verifyDBSCAN(space, points, 3 * r, 3,
                            toView<DeviceType, int>({5, 5, 5, 5})));
    BOOST_TEST(verifyDBSCAN(space, points, 3 * r, 4,
                            toView<DeviceType, int>({7, 7, 7, 7})));
    BOOST_TEST(verifyDBSCAN(space, points, 3 * r, 5,
                            toView<DeviceType, int>({-1, -1, -1, -1})));
  }

  {
    // check for bridging effect
    auto points = toView<DeviceType, Point>({{-1, 0.5, 0},
                                             {-1, -0.5, 0},
                                             {-1, 0, 0},
                                             {{0, 0, 0}},
                                             {{1, 0, 0}},
                                             {{1, 0.5, 0}},
                                             {{1, -0.5, 0}}});

    BOOST_TEST(verifyDBSCAN(space, points, 1, 3,
                            toView<DeviceType, int>({5, 5, 5, 5, 5, 5, 5})));
    BOOST_TEST((verifyDBSCAN(space, points, 1, 4,
                             toView<DeviceType, int>({5, 5, 5, 5, 6, 6, 6})) ||
                verifyDBSCAN(space, points, 1, 4,
                             toView<DeviceType, int>({5, 5, 5, 6, 6, 6, 6}))));
    BOOST_TEST(!verifyDBSCAN(space, points, 1, 4,
                             toView<DeviceType, int>({5, 5, 5, 5, 5, 5, 5})));
  }

  {
    // check where a core point is connected to only boundary points, but which
    // are stripped by a second core point

    // o - core, x - border
    //     -1 0 1 2
    //    ----------
    //  2 | x o x
    //  1 |   x   x
    //  0 |   o x o
    // -1 |   x   x
    // -2 | x o x
    // clang-format off
    auto points = toView<DeviceType, Point>({
        {0, -2, 0}, {-1, -2, 0}, {1, -2, 0}, {0, -1, 0}, // bottom
        {0, 2, 0}, {-1, 2, 0}, {1, 2, 0}, {0, 1, 0}, // top
        {2, 0, 0}, {2, -1, 0}, {2, 1, 0}, {1, 0, 0}, // right
        {0, 0, 0}  // stripped core
    });
    // clang-format on

    BOOST_TEST(verifyDBSCAN(
        space, points, 1, 4,
        toView<DeviceType, int>({0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 5})));
    // make sure the stripped core is not marked as noise
    BOOST_TEST(!verifyDBSCAN(
        space, points, 1, 4,
        toView<DeviceType, int>({0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, -1})));
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(dbscan, DeviceType, ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using ArborX::dbscan;
  using ArborX::Point;
  using ArborX::Details::verifyDBSCAN;

  ExecutionSpace space;

  for (auto impl : {ArborX::DBSCAN::Implementation::FDBSCAN,
                    ArborX::DBSCAN::Implementation::FDBSCAN_DenseBox})
  {
    ArborX::DBSCAN::Parameters params;
    params.setImplementation(impl);
    {
      auto points = toView<DeviceType, Point>({{{0, 0, 0}}, {{1, 1, 1}}});

      auto r = std::sqrt(3);

      BOOST_TEST(verifyDBSCAN(space, points, r - 0.1, 2,
                              dbscan(space, points, r - 0.1, 2, params)));
      BOOST_TEST(verifyDBSCAN(space, points, r, 2,
                              dbscan(space, points, r, 2, params)));
      BOOST_TEST(verifyDBSCAN(space, points, r, 3,
                              dbscan(space, points, r, 3, params)));

      // Test non-View primitives
      HiddenView<decltype(points)> hidden_points{points};
      BOOST_TEST(
          verifyDBSCAN(space, hidden_points, r - 0.1, 2,
                       dbscan(space, hidden_points, r - 0.1, 2, params)));
      BOOST_TEST(verifyDBSCAN(space, hidden_points, r, 2,
                              dbscan(space, hidden_points, r, 2, params)));
      BOOST_TEST(verifyDBSCAN(space, hidden_points, r, 3,
                              dbscan(space, hidden_points, r, 3, params)));
    }

    {
      auto points = toView<DeviceType, Point>(
          {{{0, 0, 0}}, {{1, 1, 1}}, {{3, 3, 3}}, {{6, 6, 6}}});

      auto r = std::sqrt(3);

      BOOST_TEST(verifyDBSCAN(space, points, r, 2,
                              dbscan(space, points, r, 2, params)));
      BOOST_TEST(verifyDBSCAN(space, points, r, 3,
                              dbscan(space, points, r, 3, params)));

      BOOST_TEST(verifyDBSCAN(space, points, 2 * r, 2,
                              dbscan(space, points, 2 * r, 2, params)));
      BOOST_TEST(verifyDBSCAN(space, points, 2 * r, 3,
                              dbscan(space, points, 2 * r, 3, params)));
      BOOST_TEST(verifyDBSCAN(space, points, 2 * r, 4,
                              dbscan(space, points, 2 * r, 4, params)));

      BOOST_TEST(verifyDBSCAN(space, points, 3 * r, 2,
                              dbscan(space, points, 3 * r, 2, params)));
      BOOST_TEST(verifyDBSCAN(space, points, 3 * r, 3,
                              dbscan(space, points, 3 * r, 3, params)));
      BOOST_TEST(verifyDBSCAN(space, points, 3 * r, 4,
                              dbscan(space, points, 3 * r, 4, params)));
      BOOST_TEST(verifyDBSCAN(space, points, 3 * r, 5,
                              dbscan(space, points, 3 * r, 5, params)));
    }

    {
      // check for bridging effect
      auto points = toView<DeviceType, Point>({{-1, 0.5, 0},
                                               {-1, -0.5, 0},
                                               {-1, 0, 0},
                                               {{0, 0, 0}},
                                               {{1, 0, 0}},
                                               {{1, 0.5, 0}},
                                               {{1, -0.5, 0}}});

      BOOST_TEST(verifyDBSCAN(space, points, 1.0, 3,
                              dbscan(space, points, 1, 3, params)));
      BOOST_TEST(verifyDBSCAN(space, points, 1.0, 4,
                              dbscan(space, points, 1, 4, params)));
    }
  }

  {
    // check where a core point is connected to only boundary points, but which
    // are stripped by a second core point

    // o - core, x - border
    //     -1 0 1 2
    //    ----------
    //  2 | x o x
    //  1 |   x   x
    //  0 |   o x o
    // -1 |   x   x
    // -2 | x o x
    // clang-format off
    auto points = toView<DeviceType, Point>({
        {0, -2, 0}, {-1, -2, 0}, {1, -2, 0}, {0, -1, 0}, // bottom
        {0, 2, 0}, {-1, 2, 0}, {1, 2, 0}, {0, 1, 0}, // top
        {2, 0, 0}, {2, -1, 0}, {2, 1, 0}, {1, 0, 0}, // right
        {0, 0, 0}  // stripped core
    });
    // clang-format on

    // This does *not* guarantee to trigger the issue, as it depends on the
    // specific implementation and runtime. But it may.
    BOOST_TEST(verifyDBSCAN(space, points, 1, 4, dbscan(space, points, 1, 4)));
  }
}

BOOST_AUTO_TEST_SUITE_END()
