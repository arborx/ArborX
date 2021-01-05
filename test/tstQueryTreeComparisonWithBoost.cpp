/****************************************************************************
 * Copyright (c) 2012-2021 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include "ArborX_BoostRTreeHelpers.hpp"
#include "ArborX_EnableDeviceTypes.hpp" // ARBORX_DEVICE_TYPES
#include <ArborX_LinearBVH.hpp>

#include <boost/test/unit_test.hpp>

#include <array>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

#include "Search_UnitTestHelpers.hpp"

BOOST_AUTO_TEST_SUITE(ComparisonWithBoost)

namespace tt = boost::test_tools;

std::vector<std::array<double, 3>>
make_stuctured_cloud(double Lx, double Ly, double Lz, int nx, int ny, int nz)
{
  std::vector<std::array<double, 3>> cloud(nx * ny * nz);
  std::function<int(int, int, int)> ind = [nx, ny](int i, int j, int k) {
    return i + j * nx + k * (nx * ny);
  };
  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j)
      for (int k = 0; k < nz; ++k)
      {
        cloud[ind(i, j, k)] = {
            {i * Lx / (nx - 1), j * Ly / (ny - 1), k * Lz / (nz - 1)}};
      }
  return cloud;
}

std::vector<std::array<double, 3>> make_random_cloud(double Lx, double Ly,
                                                     double Lz, int n)
{
  std::vector<std::array<double, 3>> cloud(n);
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution_x(0.0, Lx);
  std::uniform_real_distribution<double> distribution_y(0.0, Ly);
  std::uniform_real_distribution<double> distribution_z(0.0, Lz);
  for (int i = 0; i < n; ++i)
  {
    double x = distribution_x(generator);
    double y = distribution_y(generator);
    double z = distribution_z(generator);
    cloud[i] = {{x, y, z}};
  }
  return cloud;
}

BOOST_AUTO_TEST_CASE_TEMPLATE(boost_rtree, DeviceType, ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;

  // construct a cloud of points (nodes of a structured grid)
  double Lx = 10.0;
  double Ly = 10.0;
  double Lz = 10.0;
  int nx = 11;
  int ny = 11;
  int nz = 11;
  auto cloud = make_stuctured_cloud(Lx, Ly, Lz, nx, ny, nz);
  int n = cloud.size();

  Kokkos::View<ArborX::Box *, DeviceType> bounding_boxes("bounding_boxes", n);
  auto bounding_boxes_host = Kokkos::create_mirror_view(bounding_boxes);
  // build bounding volume hierarchy
  for (int i = 0; i < n; ++i)
  {
    auto const &point = cloud[i];
    double x = std::get<0>(point);
    double y = std::get<1>(point);
    double z = std::get<2>(point);
    bounding_boxes_host[i] = {{{x, y, z}}, {{x, y, z}}};
  }

  Kokkos::deep_copy(bounding_boxes, bounding_boxes_host);

  // random points for radius search and kNN queries
  // compare our solution against Boost R-tree
  int const n_points = 100;
  auto queries = make_random_cloud(Lx, Ly, Lz, n_points);
  Kokkos::View<double * [3], ExecutionSpace> point_coords("point_coords",
                                                          n_points);
  auto point_coords_host = Kokkos::create_mirror_view(point_coords);
  Kokkos::View<double *, ExecutionSpace> radii("radii", n_points);
  auto radii_host = Kokkos::create_mirror_view(radii);
  Kokkos::View<int * [2], ExecutionSpace> within_n_pts("within_n_pts",
                                                       n_points);
  Kokkos::View<int * [2], ExecutionSpace> nearest_n_pts("nearest_n_pts",
                                                        n_points);
  Kokkos::View<int *, ExecutionSpace> k("distribution_k", n_points);
  auto k_host = Kokkos::create_mirror_view(k);
  // use random radius for the search and random number k of for the kNN
  // search
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution_radius(
      0.0, std::sqrt(Lx * Lx + Ly * Ly + Lz * Lz));
  std::uniform_int_distribution<int> distribution_k(
      1, std::floor(sqrt(nx * nx + ny * ny + nz * nz)));
  for (unsigned int i = 0; i < n_points; ++i)
  {
    auto const &point = queries[i];
    double x = std::get<0>(point);
    double y = std::get<1>(point);
    double z = std::get<2>(point);
    radii_host[i] = distribution_radius(generator);
    k_host[i] = distribution_k(generator);
    point_coords_host(i, 0) = x;
    point_coords_host(i, 1) = y;
    point_coords_host(i, 2) = z;
  }

  Kokkos::deep_copy(point_coords, point_coords_host);
  Kokkos::deep_copy(radii, radii_host);
  Kokkos::deep_copy(k, k_host);

  Kokkos::View<ArborX::Nearest<ArborX::Point> *, DeviceType> nearest_queries(
      "nearest_queries", n_points);
  Kokkos::parallel_for(
      "register_nearest_queries",
      Kokkos::RangePolicy<ExecutionSpace>(0, n_points), KOKKOS_LAMBDA(int i) {
        nearest_queries(i) = ArborX::nearest<ArborX::Point>(
            {{point_coords(i, 0), point_coords(i, 1), point_coords(i, 2)}},
            k(i));
      });
  auto nearest_queries_host = Kokkos::create_mirror_view(nearest_queries);
  Kokkos::deep_copy(nearest_queries_host, nearest_queries);

  Kokkos::View<decltype(ArborX::intersects(ArborX::Sphere{})) *, DeviceType>
      within_queries("within_queries", n_points);
  Kokkos::parallel_for(
      "register_within_queries",
      Kokkos::RangePolicy<ExecutionSpace>(0, n_points), KOKKOS_LAMBDA(int i) {
        within_queries(i) = ArborX::intersects(ArborX::Sphere{
            {{point_coords(i, 0), point_coords(i, 1), point_coords(i, 2)}},
            radii(i)});
      });
  auto within_queries_host = Kokkos::create_mirror_view(within_queries);
  Kokkos::deep_copy(within_queries_host, within_queries);

  ArborX::BVH<typename DeviceType::memory_space> bvh(ExecutionSpace{},
                                                     bounding_boxes);

  BoostExt::RTree<ArborX::Box> rtree(ExecutionSpace{}, bounding_boxes_host);

  // FIXME check currently sporadically fails when using the HIP backend
  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, bvh, nearest_queries,
                         query(ExecutionSpace{}, rtree, nearest_queries_host));

  // FIXME ditto
  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, bvh, within_queries,
                         query(ExecutionSpace{}, rtree, within_queries_host));
}

BOOST_AUTO_TEST_SUITE_END()
