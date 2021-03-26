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

inline std::vector<ArborX::Point>
make_stuctured_cloud(double Lx, double Ly, double Lz, int nx, int ny, int nz)
{
  std::function<int(int, int, int)> ind = [nx, ny](int i, int j, int k) {
    return i + j * nx + k * (nx * ny);
  };
  std::vector<ArborX::Point> cloud(nx * ny * nz);
  for (int i = 0; i < nx; ++i)
    for (int j = 0; j < ny; ++j)
      for (int k = 0; k < nz; ++k)
        cloud[ind(i, j, k)] = {
            {i * Lx / (nx - 1), j * Ly / (ny - 1), k * Lz / (nz - 1)}};
  return cloud;
}

template <typename Geometry>
std::vector<Geometry> make_random_cloud(double Lx, double Ly, double Lz, int n,
                                        int seed = 0);

template <>
inline std::vector<ArborX::Point> make_random_cloud(double Lx, double Ly,
                                                    double Lz, int n, int seed)
{
  std::default_random_engine generator(seed);
  std::uniform_real_distribution<double> distribution_x(0.0, Lx);
  std::uniform_real_distribution<double> distribution_y(0.0, Ly);
  std::uniform_real_distribution<double> distribution_z(0.0, Lz);

  std::vector<ArborX::Point> cloud;
  cloud.reserve(n);
  for (int i = 0; i < n; ++i)
  {
    double x = distribution_x(generator);
    double y = distribution_y(generator);
    double z = distribution_z(generator);
    cloud.emplace_back(x, y, z);
  }
  return cloud;
}

template <>
inline std::vector<ArborX::Box> make_random_cloud(double Lx, double Ly,
                                                  double Lz, int n, int seed)
{
  auto const min_xyz = std::min(std::min(Lx, Ly), Lz);
  std::default_random_engine generator(seed);
  std::uniform_real_distribution<float> distribution_l(
      0.0, min_xyz / (float)n); // divide by n to avoid too many overlaps

  auto point_cloud = make_random_cloud<ArborX::Point>(Lx, Ly, Lz, n, seed);
  std::vector<ArborX::Box> cloud;
  cloud.reserve(n);
  for (int i = 0; i < n; ++i)
  {
    float length = distribution_l(generator);
    auto const &min_corner = point_cloud[i];
    ArborX::Point max_corner{min_corner[0] + length, min_corner[1] + length,
                             min_corner[2] + length};
    cloud.emplace_back(min_corner, max_corner);
  }
  return cloud;
}

template <typename Tree, typename DeviceType, typename PrimitiveGeometry,
          typename PredicateGeometry>
void test_spatial_predicate(std::vector<PrimitiveGeometry> const &sources,
                            std::vector<PredicateGeometry> const &targets,
                            std::vector<float> const &radii)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;

  assert(targets.size() == radii.size());

  Kokkos::View<PrimitiveGeometry *, DeviceType> primitives(
      "Testing::primitives", sources.size());
  auto primitives_host = Kokkos::create_mirror_view(primitives);
  Kokkos::deep_copy(primitives_host,
                    Kokkos::View<PrimitiveGeometry const *, Kokkos::HostSpace,
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>{
                        sources.data(), sources.size()});
  Kokkos::deep_copy(primitives, primitives_host);

  auto const n_queries = targets.size();
  Kokkos::View<decltype(ArborX::intersects(ArborX::Sphere{})) *, DeviceType>
      within_queries("Testing::within_queries", n_queries);
  auto within_queries_host = Kokkos::create_mirror_view(within_queries);
  for (int i = 0; i < (int)n_queries; ++i)
    within_queries_host(i) =
        ArborX::intersects(ArborX::Sphere{targets[i], radii[i]});
  Kokkos::deep_copy(within_queries, within_queries_host);

  Tree tree(ExecutionSpace{},
            Kokkos::create_mirror_view_and_copy(MemorySpace{}, primitives));

  BoostExt::RTree<PrimitiveGeometry> rtree(ExecutionSpace{}, primitives_host);

  // FIXME check currently sporadically fails when using the HIP backend
  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree, within_queries,
                         query(ExecutionSpace{}, rtree, within_queries_host));
}

template <typename Tree, typename DeviceType, typename PrimitiveGeometry,
          typename PredicateGeometry>
void test_nearest_predicate(std::vector<PrimitiveGeometry> const &sources,
                            std::vector<PredicateGeometry> const &targets,
                            std::vector<int> const &ks)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;

  assert(targets.size() == ks.size());

  Kokkos::View<PrimitiveGeometry *, DeviceType> primitives(
      "Testing::primitives", sources.size());
  auto primitives_host = Kokkos::create_mirror_view(primitives);
  Kokkos::deep_copy(primitives_host,
                    Kokkos::View<PrimitiveGeometry const *, Kokkos::HostSpace,
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>{
                        sources.data(), sources.size()});
  Kokkos::deep_copy(primitives, primitives_host);

  auto const n_queries = targets.size();
  Kokkos::View<ArborX::Nearest<PredicateGeometry> *, DeviceType>
      nearest_queries("Testing::nearest_queries", n_queries);
  auto nearest_queries_host = Kokkos::create_mirror_view(nearest_queries);
  for (int i = 0; i < (int)n_queries; ++i)
    nearest_queries_host(i) = ArborX::nearest(targets[i], ks[i]);
  Kokkos::deep_copy(nearest_queries, nearest_queries_host);

  Tree tree(ExecutionSpace{},
            Kokkos::create_mirror_view_and_copy(MemorySpace{}, primitives));

  BoostExt::RTree<PrimitiveGeometry> rtree(ExecutionSpace{}, primitives_host);

  // FIXME check currently sporadically fails when using the HIP backend
  ARBORX_TEST_QUERY_TREE(ExecutionSpace{}, tree, nearest_queries,
                         query(ExecutionSpace{}, rtree, nearest_queries_host));
}

// FIXME temporary workaround bug in HIP-Clang (register spill)
#ifdef KOKKOS_ENABLE_HIP
BOOST_TEST_DECORATOR(*boost::unit_test::expected_failures(1))
#endif
BOOST_AUTO_TEST_CASE_TEMPLATE(boost_rtree_spatial_predicate, TreeTypeTraits,
                              TreeTypeTraitsList)
{
  using Tree = typename TreeTypeTraits::type;
  using DeviceType = typename TreeTypeTraits::device_type;

  double Lx = 10.0;
  double Ly = 10.0;
  double Lz = 10.0;
  int nx = 11;
  int ny = 11;
  int nz = 11;
  int const n_queries = 100;

  {
    auto sources = make_stuctured_cloud(Lx, Ly, Lz, nx, ny, nz);
    auto targets = make_random_cloud<ArborX::Point>(Lx, Ly, Lz, n_queries);

    // use random number k of for the kNN search
    std::vector<float> radii(n_queries);
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution_radius(
        0.f, std::sqrt(Lx * Lx + Ly * Ly + Lz * Lz));
    for (unsigned int i = 0; i < n_queries; ++i)
      radii[i] = distribution_radius(generator);

    test_spatial_predicate<Tree, DeviceType>(sources, targets, radii);
  }
}

#ifndef ARBORX_TEST_DISABLE_NEAREST_QUERY
// FIXME temporary workaround bug in HIP-Clang (register spill)
#ifdef KOKKOS_ENABLE_HIP
BOOST_TEST_DECORATOR(*boost::unit_test::expected_failures(1))
#endif
BOOST_AUTO_TEST_CASE_TEMPLATE(boost_rtree_nearest_predicate, TreeTypeTraits,
                              TreeTypeTraitsList)
{
  using Tree = typename TreeTypeTraits::type;
  using DeviceType = typename TreeTypeTraits::device_type;

  double Lx = 10.0;
  double Ly = 10.0;
  double Lz = 10.0;
  int nx = 11;
  int ny = 11;
  int nz = 11;
  int const n_queries = 100;

  {
    auto sources = make_stuctured_cloud(Lx, Ly, Lz, nx, ny, nz);
    auto targets = make_random_cloud<ArborX::Point>(Lx, Ly, Lz, n_queries);

    // use random number k of for the kNN search
    std::vector<int> ks(n_queries);
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution_k(
        1, std::floor(sqrt(nx * nx + ny * ny + nz * nz)));
    for (unsigned int i = 0; i < n_queries; ++i)
      ks[i] = distribution_k(generator);

    test_nearest_predicate<Tree, DeviceType>(sources, targets, ks);
  }

  {
    auto sources = make_random_cloud<ArborX::Point>(Lx, Ly, Lz, nx * ny * nz);
    // Make seed the size of the point cloud for the tree. This way we
    // guarantee that the query points are different from the tree points.
    int const seed = sources.size();
    auto targets =
        make_random_cloud<ArborX::Point>(Lx, Ly, Lz, n_queries, seed);

    std::vector<int> ks(n_queries);
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution_k(
        1, std::floor(sqrt(nx * nx + ny * ny + nz * nz)));
    for (unsigned int i = 0; i < n_queries; ++i)
    {
      // make sure that at least few have k = 1 (common use case)
      bool const use_k1 = (i == 0 || (i % 13 == 0));
      ks[i] = (!use_k1 ? distribution_k(generator) : 1);
    }

    test_nearest_predicate<Tree, DeviceType>(sources, targets, ks);
  }

  {
    auto sources = make_stuctured_cloud(Lx, Ly, Lz, nx, ny, nz);
    auto targets = make_random_cloud<ArborX::Box>(Lx, Ly, Lz, n_queries);

    // use random number k of for the kNN search
    std::vector<int> ks(n_queries);
    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution_k(
        1, std::floor(sqrt(nx * nx + ny * ny + nz * nz)));
    for (unsigned int i = 0; i < n_queries; ++i)
      ks[i] = distribution_k(generator);

    test_nearest_predicate<Tree, DeviceType>(sources, targets, ks);
  }
}
#endif

BOOST_AUTO_TEST_SUITE_END()
