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

#include "ArborX_BoostRTreeHelpers.hpp"

#include <boost/test/unit_test.hpp>

#include <functional>
#include <random>

#include "Search_UnitTestHelpers.hpp"
// clang-format off
#include "ArborXTest_TreeTypeTraits.hpp"
// clang-format on
#include <Kokkos_Core.hpp>

// We need a forward-declaration for NVCC, see below.
namespace ArborX
{
namespace Experimental
{
template <int k>
struct KDOP;
}
} // namespace ArborX

BOOST_AUTO_TEST_SUITE(ComparisonWithBoost)

namespace tt = boost::test_tools;

inline Kokkos::View<ArborX::Point *, Kokkos::HostSpace>
make_stuctured_cloud(double Lx, double Ly, double Lz, int nx, int ny, int nz)
{
  std::function<int(int, int, int)> ind = [nx, ny](int i, int j, int k) {
    return i + j * nx + k * (nx * ny);
  };
  Kokkos::View<ArborX::Point *, Kokkos::HostSpace> cloud(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "structured_cloud"),
      nx * ny * nz);
  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j)
      for (int k = 0; k < nz; ++k)
      {
        cloud[ind(i, j, k)] = {
            {i * Lx / (nx - 1), j * Ly / (ny - 1), k * Lz / (nz - 1)}};
      }
  return cloud;
}

template <typename Geometry>
Kokkos::View<Geometry *, Kokkos::HostSpace>
make_random_cloud(double Lx, double Ly, double Lz, int n);

template <>
inline Kokkos::View<ArborX::Point *, Kokkos::HostSpace>
make_random_cloud(double Lx, double Ly, double Lz, int n)
{
  Kokkos::View<ArborX::Point *, Kokkos::HostSpace> cloud(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "random_cloud"), n);
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

template <>
inline Kokkos::View<ArborX::Box *, Kokkos::HostSpace>
make_random_cloud(double Lx, double Ly, double Lz, int n)
{
  Kokkos::View<ArborX::Box *, Kokkos::HostSpace> cloud(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "random_cloud"), n);
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution_x(0.0, Lx);
  std::uniform_real_distribution<double> distribution_y(0.0, Ly);
  std::uniform_real_distribution<double> distribution_z(0.0, Lz);
  double const min_xyz = std::min(std::min(Lx, Ly), Lz);
  // We divide min_xyz by n in order to avoid a large number of overlapping
  // boxes
  std::uniform_real_distribution<double> distribution_l(
      0.0, min_xyz / static_cast<double>(n));
  for (int i = 0; i < n; ++i)
  {
    float x = distribution_x(generator);
    float y = distribution_y(generator);
    float z = distribution_z(generator);
    float length = distribution_l(generator);
    cloud[i] = {{x, y, z}, {x + length, y + length, z + length}};
  }
  return cloud;
}

template <typename Tree, typename ExecutionSpace, typename DeviceType,
          typename PrimitiveGeometry>
void boost_rtree_nearest_predicate()
{
  // construct a cloud of points (nodes of a structured grid)
  double Lx = 10.0;
  double Ly = 10.0;
  double Lz = 10.0;
  int nx = 11;
  int ny = 11;
  int nz = 11;
  auto cloud = make_stuctured_cloud(Lx, Ly, Lz, nx, ny, nz);

  // random objects for kNN queries
  // compare our solution against Boost R-tree
  int const n_queries = 100;
  using MemorySpace = typename Tree::memory_space;
  auto geometry_objects = Kokkos::create_mirror_view_and_copy(
      MemorySpace{},
      make_random_cloud<PrimitiveGeometry>(Lx, Ly, Lz, n_queries));

  Kokkos::View<int *, ExecutionSpace> k("k", n_queries);
  auto k_host = Kokkos::create_mirror_view(k);
  // use random number k of for the kNN search
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution_k(
      1, std::floor(sqrt(nx * nx + ny * ny + nz * nz)));
  for (unsigned int i = 0; i < n_queries; ++i)
  {
    k_host[i] = distribution_k(generator);
  }

  Kokkos::deep_copy(k, k_host);

  Kokkos::View<ArborX::Nearest<PrimitiveGeometry> *, DeviceType>
      nearest_queries("nearest_queries", n_queries);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace>(0, n_queries), KOKKOS_LAMBDA(int i) {
        nearest_queries(i) = ArborX::nearest(geometry_objects(i), k(i));
      });
  auto nearest_queries_host = Kokkos::create_mirror_view(nearest_queries);
  Kokkos::deep_copy(nearest_queries_host, nearest_queries);

  Tree tree(ExecutionSpace{},
            Kokkos::create_mirror_view_and_copy(MemorySpace{}, cloud));

  BoostExt::RTree<decltype(cloud)::value_type> rtree(ExecutionSpace{}, cloud);

  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree, nearest_queries,
                         query(ExecutionSpace{}, rtree, nearest_queries_host));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(boost_rtree_spatial_predicate, TreeTypeTraits,
                              TreeTypeTraitsList)
{
  using Tree = typename TreeTypeTraits::type;
  using ExecutionSpace = typename TreeTypeTraits::execution_space;
  using DeviceType = typename TreeTypeTraits::device_type;

  // FIXME_NVCC we see inexplainable test failures with NVCC and KDOP<18> and
  // KDOP<26> here.
#ifdef __NVCC__
  if (std::is_same<typename Tree::bounding_volume_type,
                   ArborX::Experimental::KDOP<18>>::value ||
      std::is_same<typename Tree::bounding_volume_type,
                   ArborX::Experimental::KDOP<26>>::value)
    return;
#endif

  // construct a cloud of points (nodes of a structured grid)
  double Lx = 10.0;
  double Ly = 10.0;
  double Lz = 10.0;
  int nx = 11;
  int ny = 11;
  int nz = 11;
  auto cloud = make_stuctured_cloud(Lx, Ly, Lz, nx, ny, nz);

  // random points for radius search
  // compare our solution against Boost R-tree
  int const n_points = 100;
  using MemorySpace = typename Tree::memory_space;
  auto points = Kokkos::create_mirror_view_and_copy(
      MemorySpace{}, make_random_cloud<ArborX::Point>(Lx, Ly, Lz, n_points));

  Kokkos::View<double *, ExecutionSpace> radii("radii", n_points);
  auto radii_host = Kokkos::create_mirror_view(radii);
  // use random radius for the search
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution_radius(
      0.0, std::sqrt(Lx * Lx + Ly * Ly + Lz * Lz));
  for (unsigned int i = 0; i < n_points; ++i)
  {
    radii_host[i] = distribution_radius(generator);
  }

  Kokkos::deep_copy(radii, radii_host);

  Kokkos::View<decltype(ArborX::intersects(ArborX::Sphere{})) *, DeviceType>
      within_queries("within_queries", n_points);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace>(0, n_points), KOKKOS_LAMBDA(int i) {
        within_queries(i) =
            ArborX::intersects(ArborX::Sphere{points(i), radii(i)});
      });
  auto within_queries_host = Kokkos::create_mirror_view(within_queries);
  Kokkos::deep_copy(within_queries_host, within_queries);

  Kokkos::View<decltype(ArborX::intersects(ArborX::Box{})) *, DeviceType>
      intersects_queries("intersects_queries", n_points);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace>(0, n_points), KOKKOS_LAMBDA(int i) {
        ArborX::Box box{{static_cast<float>(points(i)[0] - radii(i)),
                         static_cast<float>(points(i)[1] - radii(i)),
                         static_cast<float>(points(i)[2] - radii(i))},
                        {static_cast<float>(points(i)[0] + radii(i)),
                         static_cast<float>(points(i)[1] + radii(i)),
                         static_cast<float>(points(i)[2] + radii(i))}};
        intersects_queries(i) = ArborX::intersects(box);
      });
  auto intersects_queries_host = Kokkos::create_mirror_view(intersects_queries);
  Kokkos::deep_copy(intersects_queries_host, intersects_queries);

  Tree tree(ExecutionSpace{},
            Kokkos::create_mirror_view_and_copy(MemorySpace{}, cloud));

  BoostExt::RTree<decltype(cloud)::value_type> rtree(ExecutionSpace{}, cloud);

  ARBORX_TEST_QUERY_TREE(
      ExecutionSpace{}, tree, intersects_queries,
      query(ExecutionSpace{}, rtree, intersects_queries_host));
#ifndef ARBORX_TEST_DISABLE_SPATIAL_QUERY_INTERSECTS_SPHERE
  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree, within_queries,
                         query(ExecutionSpace{}, rtree, within_queries_host));
#endif
}

#ifndef ARBORX_TEST_DISABLE_NEAREST_QUERY
BOOST_AUTO_TEST_CASE_TEMPLATE(boost_rtree_nearest_predicate_point,
                              TreeTypeTraits, TreeTypeTraitsList)
{
  using Tree = typename TreeTypeTraits::type;
  using ExecutionSpace = typename TreeTypeTraits::execution_space;
  using DeviceType = typename TreeTypeTraits::device_type;

  boost_rtree_nearest_predicate<Tree, ExecutionSpace, DeviceType,
                                ArborX::Point>();
}

BOOST_AUTO_TEST_CASE_TEMPLATE(boost_rtree_nearest_predicate_box, TreeTypeTraits,
                              TreeTypeTraitsList)
{
  using Tree = typename TreeTypeTraits::type;
  using ExecutionSpace = typename TreeTypeTraits::execution_space;
  using DeviceType = typename TreeTypeTraits::device_type;

  boost_rtree_nearest_predicate<Tree, ExecutionSpace, DeviceType,
                                ArborX::Box>();
}
#endif

BOOST_AUTO_TEST_SUITE_END()
