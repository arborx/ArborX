/****************************************************************************
 * Copyright (c) 2017-2021 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
#include "ArborX_DBSCANVerification.hpp"
#include "ArborX_EnableDeviceTypes.hpp" // ARBORX_DEVICE_TYPES
#include <ArborX_DBSCAN.hpp>

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

BOOST_AUTO_TEST_SUITE(DBSCAN)

template <typename DeviceType, typename T>
auto buildView(std::vector<T> const &v)
{
  Kokkos::View<T *, DeviceType> view("Testing::v", v.size());
  Kokkos::deep_copy(view, Kokkos::View<T const *, Kokkos::HostSpace,
                                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                              v.data(), v.size()));
  return view;
}

BOOST_AUTO_TEST_CASE_TEMPLATE(dbscan_verifier, DeviceType, ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using ArborX::Point;
  using ArborX::Details::verifyDBSCAN;

  ExecutionSpace space;

  {
    auto points = buildView<DeviceType, Point>({{{0, 0, 0}}, {{1, 1, 1}}});

    auto r = std::sqrt(3);

    BOOST_TEST(verifyDBSCAN(space, points, r - 0.1, 2,
                            buildView<DeviceType, int>({-1, -1})));
    BOOST_TEST(!verifyDBSCAN(space, points, r - 0.1, 2,
                             buildView<DeviceType, int>({1, 2})));
    BOOST_TEST(!verifyDBSCAN(space, points, r - 0.1, 2,
                             buildView<DeviceType, int>({1, 1})));
    BOOST_TEST(
        verifyDBSCAN(space, points, r, 2, buildView<DeviceType, int>({1, 1})));
    BOOST_TEST(
        !verifyDBSCAN(space, points, r, 2, buildView<DeviceType, int>({1, 2})));
    BOOST_TEST(verifyDBSCAN(space, points, r, 3,
                            buildView<DeviceType, int>({-1, -1})));
    BOOST_TEST(
        !verifyDBSCAN(space, points, r, 3, buildView<DeviceType, int>({1, 1})));
  }

  {
    auto points = buildView<DeviceType, Point>(
        {{{0, 0, 0}}, {{1, 1, 1}}, {{3, 3, 3}}, {{6, 6, 6}}});

    auto r = std::sqrt(3);

    BOOST_TEST(verifyDBSCAN(space, points, r, 2,
                            buildView<DeviceType, int>({1, 1, -1, -1})));
    BOOST_TEST(verifyDBSCAN(space, points, r, 3,
                            buildView<DeviceType, int>({-1, -1, -1, -1})));

    BOOST_TEST(verifyDBSCAN(space, points, 2 * r, 2,
                            buildView<DeviceType, int>({3, 3, 3, -1})));
    BOOST_TEST(verifyDBSCAN(space, points, 2 * r, 3,
                            buildView<DeviceType, int>({3, 3, 3, -1})));
    BOOST_TEST(verifyDBSCAN(space, points, 2 * r, 4,
                            buildView<DeviceType, int>({-1, -1, -1, -1})));

    BOOST_TEST(verifyDBSCAN(space, points, 3 * r, 2,
                            buildView<DeviceType, int>({5, 5, 5, 5})));
    BOOST_TEST(verifyDBSCAN(space, points, 3 * r, 3,
                            buildView<DeviceType, int>({5, 5, 5, 5})));
    BOOST_TEST(verifyDBSCAN(space, points, 3 * r, 4,
                            buildView<DeviceType, int>({7, 7, 7, 7})));
    BOOST_TEST(verifyDBSCAN(space, points, 3 * r, 5,
                            buildView<DeviceType, int>({-1, -1, -1, -1})));
  }

  {
    // check for bridging effect
    auto points = buildView<DeviceType, Point>({{-1, 0.5, 0},
                                                {-1, -0.5, 0},
                                                {-1, 0, 0},
                                                {{0, 0, 0}},
                                                {{1, 0, 0}},
                                                {{1, 0.5, 0}},
                                                {{1, -0.5, 0}}});

    BOOST_TEST(verifyDBSCAN(space, points, 1, 3,
                            buildView<DeviceType, int>({5, 5, 5, 5, 5, 5, 5})));
    BOOST_TEST(
        (verifyDBSCAN(space, points, 1, 4,
                      buildView<DeviceType, int>({5, 5, 5, 5, 6, 6, 6})) ||
         verifyDBSCAN(space, points, 1, 4,
                      buildView<DeviceType, int>({5, 5, 5, 6, 6, 6, 6}))));
    BOOST_TEST(
        !verifyDBSCAN(space, points, 1, 4,
                      buildView<DeviceType, int>({5, 5, 5, 5, 5, 5, 5})));
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(dbscan, DeviceType, ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using ArborX::dbscan;
  using ArborX::Point;
  using ArborX::Details::verifyDBSCAN;

  ExecutionSpace space;

  {
    auto points = buildView<DeviceType, Point>({{{0, 0, 0}}, {{1, 1, 1}}});

    auto r = std::sqrt(3);

    BOOST_TEST(verifyDBSCAN(space, points, r - 0.1, 2,
                            dbscan(space, points, r - 0.1, 2)));
    BOOST_TEST(verifyDBSCAN(space, points, r, 2, dbscan(space, points, r, 2)));
    BOOST_TEST(verifyDBSCAN(space, points, r, 3, dbscan(space, points, r, 3)));

    // Test non-View primitives
    HiddenView<decltype(points)> hidden_points{points};
    BOOST_TEST(verifyDBSCAN(space, hidden_points, r - 0.1, 2,
                            dbscan(space, hidden_points, r - 0.1, 2)));
    BOOST_TEST(verifyDBSCAN(space, hidden_points, r, 2,
                            dbscan(space, hidden_points, r, 2)));
    BOOST_TEST(verifyDBSCAN(space, hidden_points, r, 3,
                            dbscan(space, hidden_points, r, 3)));
  }

  {
    auto points = buildView<DeviceType, Point>(
        {{{0, 0, 0}}, {{1, 1, 1}}, {{3, 3, 3}}, {{6, 6, 6}}});

    auto r = std::sqrt(3);

    BOOST_TEST(verifyDBSCAN(space, points, r, 2, dbscan(space, points, r, 2)));
    BOOST_TEST(verifyDBSCAN(space, points, r, 3, dbscan(space, points, r, 3)));

    BOOST_TEST(
        verifyDBSCAN(space, points, 2 * r, 2, dbscan(space, points, 2 * r, 2)));
    BOOST_TEST(
        verifyDBSCAN(space, points, 2 * r, 3, dbscan(space, points, 2 * r, 3)));
    BOOST_TEST(
        verifyDBSCAN(space, points, 2 * r, 4, dbscan(space, points, 2 * r, 4)));

    BOOST_TEST(
        verifyDBSCAN(space, points, 3 * r, 2, dbscan(space, points, 3 * r, 2)));
    BOOST_TEST(
        verifyDBSCAN(space, points, 3 * r, 3, dbscan(space, points, 3 * r, 3)));
    BOOST_TEST(
        verifyDBSCAN(space, points, 3 * r, 4, dbscan(space, points, 3 * r, 4)));
    BOOST_TEST(
        verifyDBSCAN(space, points, 3 * r, 5, dbscan(space, points, 3 * r, 5)));
  }

  {
    // check for bridging effect
    auto points = buildView<DeviceType, Point>({{-1, 0.5, 0},
                                                {-1, -0.5, 0},
                                                {-1, 0, 0},
                                                {{0, 0, 0}},
                                                {{1, 0, 0}},
                                                {{1, 0.5, 0}},
                                                {{1, -0.5, 0}}});

    BOOST_TEST(
        verifyDBSCAN(space, points, 1.0, 3, dbscan(space, points, 1, 3)));
    BOOST_TEST(
        verifyDBSCAN(space, points, 1.0, 4, dbscan(space, points, 1, 4)));
  }
}

BOOST_AUTO_TEST_SUITE_END()
