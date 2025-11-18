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
struct ArborX::AccessTraits<HiddenView<View>>
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

template <typename DeviceType, typename Coordinate>
void dbscan_verifier_f()
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using Point = ArborX::Point<3, Coordinate>;
  using ArborX::Details::verifyDBSCAN;

  ExecutionSpace space;

  auto const dbscan_verifier_result =
      [&space](auto const &points, Coordinate eps, int minpts,
               std::vector<int> const &result, std::string const &algorithm) {
        return verifyDBSCAN(space, points, eps, minpts,
                            toView<DeviceType, int>(result), algorithm);
      };

  {
    auto points = toView<DeviceType, Point>({{{0, 0, 0}}, {{1, 1, 1}}});

    Coordinate r = std::sqrt(3);

    // clang-format off
    BOOST_TEST(dbscan_verifier_result(points, r - (Coordinate)0.1, 2, {-1, -1}, "dbscan"));
    BOOST_TEST(dbscan_verifier_result(points, r - (Coordinate)0.1, 2, {-1, -1}, "dbscan*"));
    BOOST_TEST(!dbscan_verifier_result(points, r - (Coordinate)0.1, 2, {1, 2}, "dbscan"));
    BOOST_TEST(!dbscan_verifier_result(points, r - (Coordinate)0.1, 2, {1, 1}, "dbscan*"));
    BOOST_TEST(dbscan_verifier_result(points, r, 2, {1, 1}, "dbscan"));
    BOOST_TEST(dbscan_verifier_result(points, r, 2, {1, 1}, "dbscan*"));
    BOOST_TEST(!dbscan_verifier_result(points, r, 2, {1, 2}, "dbscan"));
    BOOST_TEST(!dbscan_verifier_result(points, r, 2, {1, 2}, "dbscan*"));
    BOOST_TEST(dbscan_verifier_result(points, r, 3, {-1, -1}, "dbscan"));
    BOOST_TEST(dbscan_verifier_result(points, r, 3, {-1, -1}, "dbscan*"));
    BOOST_TEST(!dbscan_verifier_result(points, r, 3, {1, 1}, "dbscan"));
    BOOST_TEST(!dbscan_verifier_result(points, r, 3, {1, 1}, "dbscan*"));
    // clang-format off
  }

  {
    auto points = toView<DeviceType, Point>(
        {{{0, 0, 0}}, {{1, 1, 1}}, {{3, 3, 3}}, {{6, 6, 6}}});

    Coordinate r = std::sqrt(3);
    Coordinate r2 = std::sqrt(12);
    Coordinate r3 = std::sqrt(48);

    BOOST_TEST(dbscan_verifier_result(points, r, 2, {1, 1, -1, -1}, "dbscan"));
    BOOST_TEST(dbscan_verifier_result(points, r, 2, {1, 1, -1, -1}, "dbscan*"));
    BOOST_TEST(dbscan_verifier_result(points, r, 3, {-1, -1, -1, -1}, "dbscan"));
    BOOST_TEST(dbscan_verifier_result(points, r, 3, {-1, -1, -1, -1}, "dbscan*"));
    BOOST_TEST(dbscan_verifier_result(points, r2, 2, {3, 3, 3, -1}, "dbscan"));
    BOOST_TEST(dbscan_verifier_result(points, r2, 2, {3, 3, 3, -1}, "dbscan*"));
    BOOST_TEST(dbscan_verifier_result(points, r2, 3, {3, 3, 3, -1}, "dbscan"));
    BOOST_TEST(dbscan_verifier_result(points, r2, 3, {-1, 3, -1, -1}, "dbscan*"));
    BOOST_TEST(dbscan_verifier_result(points, r2, 4, {-1, -1, -1, -1}, "dbscan"));
    BOOST_TEST(dbscan_verifier_result(points, r2, 4, {-1, -1, -1, -1}, "dbscan*"));
    BOOST_TEST(dbscan_verifier_result(points, r3, 2, {5, 5, 5, 5}, "dbscan"));
    BOOST_TEST(dbscan_verifier_result(points, r3, 2, {5, 5, 5, 5}, "dbscan*"));
    BOOST_TEST(dbscan_verifier_result(points, r3, 3, {5, 5, 5, 5}, "dbscan"));
    BOOST_TEST(dbscan_verifier_result(points, r3, 3, {5, 5, 5, -1}, "dbscan*"));
    BOOST_TEST(dbscan_verifier_result(points, r3, 4, {7, 7, 7, 7}, "dbscan"));
    BOOST_TEST(dbscan_verifier_result(points, r3, 4, {-1, -1, 7, -1}, "dbscan*"));
    BOOST_TEST(dbscan_verifier_result(points, r3, 5, {-1, -1, -1, -1}, "dbscan"));
    BOOST_TEST(dbscan_verifier_result(points, r3, 5, {-1, -1, -1, -1}, "dbscan*"));
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

    BOOST_TEST(dbscan_verifier_result(points, 1, 3, {5, 5, 5, 5, 5, 5, 5}, "dbscan"));
    BOOST_TEST(dbscan_verifier_result(points, 1, 3, {5, 5, 5, 5, 5, 5, 5}, "dbscan*"));
    BOOST_TEST((dbscan_verifier_result(points, 1, 4, {5, 5, 5, 5, 6, 6, 6}, "dbscan") ||
               dbscan_verifier_result(points, 1, 4, {5, 5, 5, 6, 6, 6, 6}, "dbscan")));
    BOOST_TEST(dbscan_verifier_result(points, 1, 4, {-1, -1, 5, -1, 6, -1, -1}, "dbscan*"));
  }

  {
    // check where a core point is connected to only border points, but which
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

    BOOST_TEST(dbscan_verifier_result(
        points, 1, 4, {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 5}, "dbscan"));
    BOOST_TEST(dbscan_verifier_result(
        points, 1, 4, {0, -1, -1, -1, 1, -1, -1, -1, 2, -1, -1, -1, 5},
        "dbscan*"));
    // make sure the stripped core is not marked as noise
    BOOST_TEST(!dbscan_verifier_result(
        points, 1, 4, {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, -1}, "dbscan"));
    BOOST_TEST(!dbscan_verifier_result(
        points, 1, 4, {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, -1}, "dbscan*"));
  }
#undef DBSCAN_VERIFIER_PASS
#undef DBSCAN_VERIFIER_PASS2
#undef DBSCAN_VERIFIER_FAIL
}

BOOST_AUTO_TEST_CASE_TEMPLATE(dbscan_verifier_float, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  dbscan_verifier_f<DeviceType, float>();
}

BOOST_AUTO_TEST_CASE_TEMPLATE(dbscan_verifier_double, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  dbscan_verifier_f<DeviceType, double>();
}

template <typename DeviceType, typename Coordinate>
void dbscan_f(ArborX::DBSCAN::Implementation impl, std::string const &algorithm)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;
  using ArborX::dbscan;
  using ArborX::Details::verifyDBSCAN;
  using Point = ArborX::Point<3, Coordinate>;

  ExecutionSpace space;

  ArborX::DBSCAN::Parameters params;
  params.setImplementation(impl).setAlgorithm(
      algorithm == "dbscan" ? ArborX::DBSCAN::Algorithm::DBSCAN
                            : ArborX::DBSCAN::Algorithm::DBSCAN_STAR);

  auto verify_dbscan = [&space, &params, &algorithm](
                           auto const &points, Coordinate eps, int minpts) {
    Kokkos::View<int *, MemorySpace> labels("Testing::labels", 0);
    dbscan(space, points, eps, minpts, labels, params);
    return verifyDBSCAN(space, points, eps, minpts, labels, algorithm);
  };

  {
    auto points = toView<DeviceType, Point>({{{0, 0, 0}}, {{1, 1, 1}}});

    Coordinate r = std::sqrt(3.1);

    BOOST_TEST(verify_dbscan(points, r - (Coordinate)0.1, 2));
    BOOST_TEST(verify_dbscan(points, r, 2));
    BOOST_TEST(verify_dbscan(points, r, 3));

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    // Test deprecated DBSCAN interface
    BOOST_TEST(
        verifyDBSCAN(space, points, r - (Coordinate)0.1, 2,
                     dbscan(space, points, r - (Coordinate)0.1, 2, params)));
    BOOST_TEST(
        verifyDBSCAN(space, points, r, 2, dbscan(space, points, r, 2, params)));
    BOOST_TEST(
        verifyDBSCAN(space, points, r, 3, dbscan(space, points, r, 3, params)));
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

    // Test non-View primitives
    HiddenView<decltype(points)> hidden_points{points};
    BOOST_TEST(verify_dbscan(hidden_points, r - (Coordinate)0.1, 2));
    BOOST_TEST(verify_dbscan(hidden_points, r, 2));
    BOOST_TEST(verify_dbscan(hidden_points, r, 3));
  }

  {
    auto points = toView<DeviceType, Point>(
        {{{0, 0, 0}}, {{1, 1, 1}}, {{3, 3, 3}}, {{6, 6, 6}}});

    Coordinate r = std::sqrt(3.1);

    BOOST_TEST(verify_dbscan(points, r, 2));
    BOOST_TEST(verify_dbscan(points, r, 3));

    BOOST_TEST(verify_dbscan(points, 2 * r, 2));
    BOOST_TEST(verify_dbscan(points, 2 * r, 3));
    BOOST_TEST(verify_dbscan(points, 2 * r, 4));

    BOOST_TEST(verify_dbscan(points, 3 * r, 2));
    BOOST_TEST(verify_dbscan(points, 3 * r, 3));
    BOOST_TEST(verify_dbscan(points, 3 * r, 4));
    BOOST_TEST(verify_dbscan(points, 3 * r, 5));
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

    BOOST_TEST(verify_dbscan(points, (Coordinate)1, 3));
    BOOST_TEST(verify_dbscan(points, (Coordinate)1, 4));
  }

  {
    // check where a core point is connected to only border points, but which
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
    BOOST_TEST(verify_dbscan(points, (Coordinate)1, 4));
  }

#undef VERIFY_DBSCAN
}

BOOST_AUTO_TEST_CASE_TEMPLATE(dbscan_fdbscan_float, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  dbscan_f<DeviceType, float>(ArborX::DBSCAN::Implementation::FDBSCAN,
                              "dbscan");
}
BOOST_AUTO_TEST_CASE_TEMPLATE(dbscan_fdbscan_double, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  dbscan_f<DeviceType, double>(ArborX::DBSCAN::Implementation::FDBSCAN,
                               "dbscan");
}
BOOST_AUTO_TEST_CASE_TEMPLATE(dbscan_fdbscan_densebox_float, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  dbscan_f<DeviceType, float>(ArborX::DBSCAN::Implementation::FDBSCAN_DenseBox,
                              "dbscan");
}
BOOST_AUTO_TEST_CASE_TEMPLATE(dbscan_fdbscan_densebox_double, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  dbscan_f<DeviceType, double>(ArborX::DBSCAN::Implementation::FDBSCAN_DenseBox,
                               "dbscan");
}

BOOST_AUTO_TEST_CASE_TEMPLATE(dbscan_star_fdbscan_float, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  dbscan_f<DeviceType, float>(ArborX::DBSCAN::Implementation::FDBSCAN,
                              "dbscan*");
}
BOOST_AUTO_TEST_CASE_TEMPLATE(dbscan_star_fdbscan_double, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  dbscan_f<DeviceType, double>(ArborX::DBSCAN::Implementation::FDBSCAN,
                               "dbscan*");
}
BOOST_AUTO_TEST_CASE_TEMPLATE(dbscan_star_fdbscan_densebox_float, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  dbscan_f<DeviceType, float>(ArborX::DBSCAN::Implementation::FDBSCAN_DenseBox,
                              "dbscan*");
}
BOOST_AUTO_TEST_CASE_TEMPLATE(dbscan_star_fdbscan_densebox_double, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  dbscan_f<DeviceType, double>(ArborX::DBSCAN::Implementation::FDBSCAN_DenseBox,
                               "dbscan*");
}

BOOST_AUTO_TEST_SUITE_END()
