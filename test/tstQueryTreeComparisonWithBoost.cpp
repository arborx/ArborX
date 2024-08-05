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

#include "ArborXTest_Cloud.hpp"
#include "ArborX_BoostRTreeHelpers.hpp"

#include <boost/test/unit_test.hpp>

#include <functional>
#include <random>

#include "Search_UnitTestHelpers.hpp"
// clang-format off
#include "ArborXTest_TreeTypeTraits.hpp"
// clang-format on
#include <Kokkos_Core.hpp>

BOOST_AUTO_TEST_SUITE(ComparisonWithBoost)

namespace tt = boost::test_tools;

inline Kokkos::View<ArborX::Point *, Kokkos::HostSpace>
make_stuctured_cloud(float Lx, float Ly, float Lz, int nx, int ny, int nz)
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

template <typename Tree, typename ExecutionSpace, typename DeviceType,
          typename PrimitiveGeometry>
void boost_rtree_nearest_predicate()
{
  // construct a cloud of points (nodes of a structured grid)
  float Lx = 10.0;
  float Ly = 10.0;
  float Lz = 10.0;
  int nx = 11;
  int ny = 11;
  int nz = 11;
  auto cloud = make_stuctured_cloud(Lx, Ly, Lz, nx, ny, nz);

  // random objects for kNN queries
  // compare our solution against Boost R-tree
  int const n_queries = 100;
  using MemorySpace = typename Tree::memory_space;
  auto geometry_objects = ArborXTest::make_random_cloud<PrimitiveGeometry>(
      ExecutionSpace{}, n_queries, Lx, Ly, Lz);

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
  using BoundingVolume = typename Tree::bounding_volume_type;
  if constexpr (ArborX::GeometryTraits::is_kdop_v<BoundingVolume>)
  {
    if constexpr (BoundingVolume::n_directions == 9 ||
                  BoundingVolume::n_directions == 13)
      return;
  }
#endif

  // construct a cloud of points (nodes of a structured grid)
  float Lx = 10.0;
  float Ly = 10.0;
  float Lz = 10.0;
  int nx = 11;
  int ny = 11;
  int nz = 11;
  auto cloud = make_stuctured_cloud(Lx, Ly, Lz, nx, ny, nz);

  // random points for radius search
  // compare our solution against Boost R-tree
  int const n_points = 100;
  using MemorySpace = typename Tree::memory_space;
  auto points = ArborXTest::make_random_cloud<ArborX::Point>(
      ExecutionSpace{}, n_points, Lx, Ly, Lz);

  Kokkos::View<float *, ExecutionSpace> radii("radii", n_points);
  auto radii_host = Kokkos::create_mirror_view(radii);
  // use random radius for the search
  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution_radius(
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

#ifndef ARBORX_TEST_DISABLE_SPATIAL_QUERY_INTERSECTS_BOX
  ARBORX_TEST_QUERY_TREE(
      ExecutionSpace{}, tree, intersects_queries,
      query(ExecutionSpace{}, rtree, intersects_queries_host));
#endif

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

#ifndef ARBORX_TEST_DISABLE_NEAREST_QUERY_BOX
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
#endif

BOOST_AUTO_TEST_SUITE_END()
