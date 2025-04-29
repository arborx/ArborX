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
#include "ArborXTest_StdVectorToKokkosView.hpp"
#include "ArborX_EnableDeviceTypes.hpp" // ARBORX_DEVICE_TYPES
#include "ArborX_EnableViewComparison.hpp"
#include <ArborX_Dendrogram.hpp>
#include <ArborX_MinimumSpanningTree.hpp>
#include <detail/ArborX_WeightedEdge.hpp>
#include <kokkos_ext/ArborX_KokkosExtViewHelpers.hpp>
#include <misc/ArborX_SortUtils.hpp>

#include "BoostTest_CUDA_clang_workarounds.hpp"
#include "boost_ext/TupleComparison.hpp"
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(Dendrogram)

using ArborX::Experimental::WeightedEdge;
namespace tt = boost::test_tools;

namespace
{

template <class ExecutionSpace>
auto buildDendrogram(ExecutionSpace const &exec_space,
                     std::vector<WeightedEdge> const &edges_host)
{
  using ArborXTest::toView;
  auto edges = toView<ExecutionSpace>(edges_host, "Test::edges");

  using MemorySpace = typename ExecutionSpace::memory_space;
  ArborX::Experimental::Dendrogram<MemorySpace> dendrogram{exec_space, edges};

  auto parents_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                                          dendrogram._parents);
  auto parent_heights_host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, dendrogram._parent_heights);
  return std::make_pair(parents_host, parent_heights_host);
}

} // namespace

BOOST_AUTO_TEST_CASE_TEMPLATE(dendrogram_union_find, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using ArborX::Experimental::WeightedEdge;

  ExecutionSpace space;

  {
    // Dendrogram (sorted edge indices)
    // --0--
    // |   |
    // 0   1
    auto [parents, heights] =
        buildDendrogram(space, std::vector<WeightedEdge>{{0, 1, 3.f}});
    BOOST_TEST(parents == (std::vector<int>{-1, 0, 0}), tt::per_element());
    BOOST_TEST(heights == (std::vector<float>{3.f}), tt::per_element());
  }

  {
    // Dendrogram (sorted edge indices)
    //      ----2---
    //      |      |
    //   ---1---   |
    //   |     |   |
    // --0--   |   |
    // |   |   |   |
    // 0   1   2   3
    auto [parents, heights] = buildDendrogram(
        space,
        std::vector<WeightedEdge>{{0, 3, 7.f}, {1, 2, 3.f}, {0, 1, 2.f}});
    BOOST_TEST(parents == (std::vector<int>{1, 2, -1, 0, 0, 1, 2}),
               tt::per_element());
    BOOST_TEST(heights == (std::vector<float>{2.f, 3.f, 7.f}),
               tt::per_element());
  }

  {
    // Dendrogram (sorted edge indices)
    //   ----2----
    //   |       |
    // --1--   --0--
    // |   |   |   |
    // 0   1   2   3
    auto [parents, heights] = buildDendrogram(
        space,
        std::vector<WeightedEdge>{{2, 3, 2.f}, {2, 0, 9.f}, {0, 1, 3.f}});
    BOOST_TEST(parents == (std::vector<int>{2, 2, -1, 1, 1, 0, 0}),
               tt::per_element());
    BOOST_TEST(heights == (std::vector<float>{2.f, 3.f, 9.f}),
               tt::per_element());
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(dendrogram_boruvka, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;

  using namespace ArborX::Details;

  ExecutionSpace space;

  int const n = 3000;

  // Choosing n > 5000 often results in MST producing edges of equal weight.
  // This is a bit problematic because there are multiple correspondoning
  // binary dendrograms which are all correct. This makes the comparison very
  // hard, and something we want to avoid for now. The box is changed from
  // [0,1]^3 to [0,100]^3 to decrease the chance of having equal weights.
  auto points = ArborXTest::make_random_cloud<ArborX::Point<3>>(space, n, 100.f,
                                                                100.f, 100.f);

  ArborX::Experimental::MinimumSpanningTree<MemorySpace, BoruvkaMode::HDBSCAN>
      mst(space, points);
  ArborX::Experimental::Dendrogram<MemorySpace> dendrogram(space, mst.edges);

  // Because the dendrogram in the MST is permuted, we need to reorder it in the
  // increasing edge order to compare with union-find
  auto parents_boruvka_device = KokkosExt::cloneWithoutInitializingNorCopying(
      space, mst.dendrogram_parents);
  auto heights_boruvka_device = KokkosExt::cloneWithoutInitializingNorCopying(
      space, mst.dendrogram_parent_heights);
  {
    Kokkos::View<float *, MemorySpace> weights(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                           "Testing::weights"),
        n - 1);
    Kokkos::parallel_for(
        "ArborX::Testing::compute_weights",
        Kokkos::RangePolicy(space, 0, n - 1),
        KOKKOS_LAMBDA(int i) { weights(i) = mst.edges(i).weight; });

    auto permute = sortObjects(space, weights);

    Kokkos::View<int *, MemorySpace> inv_permute(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                           "Testing::inv_permute"),
        n - 1);
    Kokkos::parallel_for(
        "Testing::compute_inv_permute", Kokkos::RangePolicy(space, 0, n - 1),
        KOKKOS_LAMBDA(int i) { inv_permute(permute(i)) = i; });

    Kokkos::parallel_for(
        "Testing::reorder_mst_dendrogram",
        Kokkos::RangePolicy(space, 0, 2 * n - 1), KOKKOS_LAMBDA(int i) {
          if (i < n - 1)
          {
            // Edge
            auto p = mst.dendrogram_parents(permute(i));
            parents_boruvka_device(i) = (p == -1 ? -1 : inv_permute(p));
            heights_boruvka_device(i) =
                mst.dendrogram_parent_heights(permute(i));
          }
          else
          {
            // Vertex
            parents_boruvka_device(i) = inv_permute(mst.dendrogram_parents(i));
          }
        });
  }
  auto parents_boruvka = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, parents_boruvka_device);
  auto heights_boruvka = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, heights_boruvka_device);

  auto parents_union_find = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, dendrogram._parents);
  auto heights_union_find = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, dendrogram._parent_heights);

  // This passes if edge weights are unique. If not, the constructed
  // dendrograms may be slightly different, as the tree may not be truly
  // binary, and converting it to binary is not unique. This could easily
  // happen for larger n or when using minPts > 1.
  BOOST_TEST(parents_boruvka == parents_union_find, tt::per_element());
  BOOST_TEST(heights_boruvka == heights_union_find, tt::per_element());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(dendrogram_boruvka_same_weights, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;

  using namespace ArborX::Details;
  using Point = ArborX::Point<2, float>;

  ExecutionSpace space;

  // Construct a Cartesian grid of points.
  // All points (except border ones) will have the same core distance.
  int const N = 4;
  int const n = N * N;
  std::vector<Point> points_v(N * N);
  for (int j = 0; j < N; ++j)
    for (int i = 0; i < N; ++i)
      points_v[j * N + i] = Point{(float)i, (float)j};
  auto points = ArborXTest::toView<DeviceType>(points_v, "Testing::points");

  // minpts = 5 is the first value that leads to the test failure with N = 4
  int const minpts = 5;
  ArborX::Experimental::MinimumSpanningTree<MemorySpace, BoruvkaMode::HDBSCAN>
      mst(space, points, minpts);
  ArborX::Experimental::Dendrogram<MemorySpace> dendrogram(space, mst.edges);

  // Check that the dendrogram is binary
  Kokkos::View<int *, MemorySpace> counts(
      Kokkos::view_alloc(space, "Testing::count"), 2 * n - 1);

  Kokkos::parallel_for(
      "Testing::count_children", Kokkos::RangePolicy(space, 0, 2 * n - 1),
      KOKKOS_LAMBDA(int i) {
        Kokkos::atomic_inc(&counts(mst.dendrogram_parents(i)));
      });

  int wrong_counts;
  Kokkos::parallel_reduce(
      "Testing::check_counts", Kokkos::RangePolicy(space, 0, 2 * n - 1),
      KOKKOS_LAMBDA(int i, int &update) {
        bool const is_edge = (i < n - 1);
        int const expected_num_children = (is_edge ? 2 : 0);
        if (counts(i) != expected_num_children)
          ++update;
      },
      wrong_counts);
  BOOST_TEST(wrong_counts == 0);
}

BOOST_AUTO_TEST_SUITE_END()
