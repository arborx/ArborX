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

template <typename Coordinate>
inline auto make_stuctured_cloud(Coordinate Lx, Coordinate Ly, Coordinate Lz,
                                 int nx, int ny, int nz)
{
  std::function<int(int, int, int)> ind = [nx, ny](int i, int j, int k) {
    return i + j * nx + k * (nx * ny);
  };
  Kokkos::View<ArborX::Point<3, Coordinate> *, Kokkos::HostSpace> cloud(
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
  using Coordinate =
      ArborX::GeometryTraits::coordinate_type_t<PrimitiveGeometry>;

  // construct a cloud of points (nodes of a structured grid)
  Coordinate Lx = 10.0;
  Coordinate Ly = 10.0;
  Coordinate Lz = 10.0;
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

  BoostExt::RTree<typename decltype(cloud)::value_type> rtree(ExecutionSpace{},
                                                              cloud);

  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree, nearest_queries,
                         query(ExecutionSpace{}, rtree, nearest_queries_host));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(boost_rtree_spatial_predicate, TreeTypeTraits,
                              TreeTypeTraitsList)
{
  using Tree = typename TreeTypeTraits::type;
  using ExecutionSpace = typename TreeTypeTraits::execution_space;
  using DeviceType = typename TreeTypeTraits::device_type;
  using BoundingVolume = typename Tree::bounding_volume_type;
  constexpr int DIM = ArborX::GeometryTraits::dimension_v<BoundingVolume>;
  using Coordinate = ArborX::GeometryTraits::coordinate_type_t<BoundingVolume>;
  using Point = ArborX::Point<DIM, Coordinate>;
  using Box = ArborX::Box<DIM, Coordinate>;

  // FIXME_NVCC we see inexplainable test failures with NVCC and KDOP<18> and
  // KDOP<26> here.
  if constexpr (ArborX::GeometryTraits::is_kdop_v<BoundingVolume>)
  {
#ifdef __NVCC__
    // FIXME_NVCC inexplicable test failures with NVCC and KDOP<18> and KDOP<26>
    if constexpr (std::is_same_v<ExecutionSpace, Kokkos::Cuda> &&
                  (BoundingVolume::n_directions == 9 ||
                   BoundingVolume::n_directions == 13))
      return;
#endif
#if defined(KOKKOS_ENABLE_SYCL) && defined(__INTEL_LLVM_COMPILER)
    // FIXME_INTEL inexplicable test failures with Intel and KDOP<14>
    if constexpr (std::is_same_v<ExecutionSpace, Kokkos::SYCL> &&
                  BoundingVolume::n_directions == 7)
      return;
#endif
  }

  // construct a cloud of points (nodes of a structured grid)
  Coordinate Lx = 10.0;
  Coordinate Ly = 10.0;
  Coordinate Lz = 10.0;
  int nx = 11;
  int ny = 11;
  int nz = 11;
  auto cloud = make_stuctured_cloud(Lx, Ly, Lz, nx, ny, nz);

  // random points for radius search
  // compare our solution against Boost R-tree
  int const n_points = 100;
  using MemorySpace = typename Tree::memory_space;
  auto points = ArborXTest::make_random_cloud<Point>(ExecutionSpace{}, n_points,
                                                     Lx, Ly, Lz);

  Kokkos::View<Coordinate *, ExecutionSpace> radii("radii", n_points);
  auto radii_host = Kokkos::create_mirror_view(radii);
  // use random radius for the search
  std::default_random_engine generator;
  std::uniform_real_distribution<Coordinate> distribution_radius(
      0.0, std::sqrt(Lx * Lx + Ly * Ly + Lz * Lz));
  for (unsigned int i = 0; i < n_points; ++i)
  {
    radii_host[i] = distribution_radius(generator);
  }

  Kokkos::deep_copy(radii, radii_host);

  using Sphere = ArborX::Sphere<DIM, Coordinate>;
  Kokkos::View<decltype(ArborX::intersects(Sphere{})) *, DeviceType>
      within_queries("within_queries", n_points);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace>(0, n_points), KOKKOS_LAMBDA(int i) {
        within_queries(i) = ArborX::intersects(Sphere{points(i), radii(i)});
      });
  auto within_queries_host = Kokkos::create_mirror_view(within_queries);
  Kokkos::deep_copy(within_queries_host, within_queries);

  Kokkos::View<decltype(ArborX::intersects(Box{})) *, DeviceType>
      intersects_queries("intersects_queries", n_points);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace>(0, n_points), KOKKOS_LAMBDA(int i) {
        Box box{{points(i)[0] - radii(i), points(i)[1] - radii(i),
                 points(i)[2] - radii(i)},
                {points(i)[0] + radii(i), points(i)[1] + radii(i),
                 points(i)[2] + radii(i)}};
        intersects_queries(i) = ArborX::intersects(box);
      });
  auto intersects_queries_host = Kokkos::create_mirror_view(intersects_queries);
  Kokkos::deep_copy(intersects_queries_host, intersects_queries);

  Tree tree(ExecutionSpace{},
            Kokkos::create_mirror_view_and_copy(MemorySpace{}, cloud));

  BoostExt::RTree<typename decltype(cloud)::value_type> rtree(ExecutionSpace{},
                                                              cloud);

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
  using BoundingVolume = typename Tree::bounding_volume_type;
  constexpr int DIM = ArborX::GeometryTraits::dimension_v<BoundingVolume>;
  using Coordinate = ArborX::GeometryTraits::coordinate_type_t<BoundingVolume>;

  boost_rtree_nearest_predicate<Tree, ExecutionSpace, DeviceType,
                                ArborX::Point<DIM, Coordinate>>();
}

BOOST_AUTO_TEST_CASE_TEMPLATE(boost_rtree_nearest_predicate_box, TreeTypeTraits,
                              TreeTypeTraitsList)
{
  using Tree = typename TreeTypeTraits::type;
  using ExecutionSpace = typename TreeTypeTraits::execution_space;
  using DeviceType = typename TreeTypeTraits::device_type;
  using BoundingVolume = typename Tree::bounding_volume_type;
  constexpr int DIM = ArborX::GeometryTraits::dimension_v<BoundingVolume>;
  using Coordinate = ArborX::GeometryTraits::coordinate_type_t<BoundingVolume>;

  boost_rtree_nearest_predicate<Tree, ExecutionSpace, DeviceType,
                                ArborX::Box<DIM, Coordinate>>();
}
#endif

BOOST_AUTO_TEST_SUITE_END()
