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

#ifndef ARBORX_SEARCH_TEST_HELPERS_HPP
#define ARBORX_SEARCH_TEST_HELPERS_HPP

// clang-format off
#include "boost_ext/KokkosPairComparison.hpp"
#include "boost_ext/TupleComparison.hpp"
#include "boost_ext/CompressedStorageComparison.hpp"
// clang-format on

#include "ArborX_EnableViewComparison.hpp"
#ifdef ARBORX_ENABLE_MPI
#include "ArborXTest_PairIndexRank.hpp"
#include "ArborX_BoostRTreeHelpers.hpp"
#include <ArborX_DistributedTree.hpp>
#endif
#include <ArborX_Box.hpp>
#include <ArborX_Point.hpp>
#include <ArborX_Sphere.hpp>

#include <boost/test/unit_test.hpp>

#include <iostream>
#include <utility>
#include <vector>

template <typename T>
struct is_distributed : std::false_type
{};

#ifdef ARBORX_ENABLE_MPI
template <typename MemorySpace, typename Value, typename... Args>
struct is_distributed<ArborX::DistributedTree<MemorySpace, Value, Args...>>
    : std::true_type
{};

template <typename I>
struct is_distributed<BoostExt::ParallelRTree<I>> : std::true_type
{};
#endif

template <typename T>
auto make_reference_solution(std::vector<T> const &values,
                             std::vector<int> const &offsets)
{
  return make_compressed_storage(offsets, values);
}

template <typename ExecutionSpace, typename Tree, typename Queries>
auto query(ExecutionSpace const &exec_space, Tree const &tree,
           Queries const &queries)
{
  using memory_space = typename Tree::memory_space;
#ifdef ARBORX_ENABLE_MPI
  using value_type = std::conditional_t<is_distributed<Tree>{},
                                        ArborXTest::PairIndexRank, int>;
#else
  using value_type = int;
#endif
  Kokkos::View<value_type *, memory_space> values("Testing::values", 0);
  Kokkos::View<int *, memory_space> offsets("Testing::offsets", 0);
  tree.query(exec_space, queries, values, offsets);
  return make_compressed_storage(
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offsets),
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, values));
}

#define ARBORX_TEST_QUERY_TREE(exec_space, tree, queries, reference)           \
  BOOST_TEST(query(exec_space, tree, queries) == (reference),                  \
             boost::test_tools::per_element());

template <typename ValueType, typename ExecutionSpace, typename Tree,
          typename Queries, typename Callback>
auto query(ExecutionSpace const &exec_space, Tree const &tree,
           Queries const &queries, Callback const &callback)
{
  using memory_space = typename Tree::memory_space;
  Kokkos::View<ValueType *, memory_space> values("Testing::values", 0);
  Kokkos::View<int *, memory_space> offsets("Testing::offsets", 0);
  tree.query(exec_space, queries, callback, values, offsets);
  return make_compressed_storage(
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offsets),
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, values));
}

#define ARBORX_TEST_QUERY_TREE_CALLBACK(exec_space, tree, queries, callback,   \
                                        reference)                             \
  using value_type = typename decltype(reference)::value_type;                 \
  BOOST_TEST(query<value_type>(exec_space, tree, queries, callback) ==         \
                 (reference),                                                  \
             boost::test_tools::per_element());

template <typename Tree, typename Geometry, typename ExecutionSpace>
auto make(ExecutionSpace const &exec_space, std::vector<Geometry> const &g)
{
  int const n = g.size();
  Kokkos::View<Geometry *, typename Tree::memory_space> geometries(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "Testing::geometries"),
      n);
  auto geometries_host =
      Kokkos::create_mirror_view(Kokkos::WithoutInitializing, geometries);
  for (int i = 0; i < n; ++i)
    geometries_host(i) = g[i];
  Kokkos::deep_copy(exec_space, geometries, geometries_host);
  return Tree(exec_space, geometries);
}

#ifdef ARBORX_ENABLE_MPI
template <typename MemorySpace, typename Geometry>
struct PairIndexRankIndexableGetter
{
  Kokkos::View<Geometry *, MemorySpace> _geometries;

  KOKKOS_FUNCTION auto const &operator()(ArborXTest::PairIndexRank p) const
  {
    return _geometries(p.index);
  }
};

template <typename DeviceType, typename Geometry>
auto makeDistributedTree(MPI_Comm comm, typename DeviceType::execution_space,
                         std::vector<Geometry> const &g)
{
  using ExecutionSpace = typename DeviceType::execution_space;
  using MemorySpace = typename DeviceType::memory_space;

  using PairIndexRank = ArborXTest::PairIndexRank;

  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);

  int const n = g.size();
  Kokkos::View<Geometry *, DeviceType> geometries("Testing::geometries", n);
  Kokkos::View<PairIndexRank *, DeviceType> pairs("Testing::geometry_pairs", n);
  auto geometries_host = Kokkos::create_mirror_view(geometries);
  auto pairs_host = Kokkos::create_mirror_view(pairs);
  for (int i = 0; i < n; ++i)
  {
    geometries_host(i) = g[i];
    pairs_host(i) = {i, comm_rank};
  }
  Kokkos::deep_copy(geometries, geometries_host);
  Kokkos::deep_copy(pairs, pairs_host);

  using IndexableGetter = PairIndexRankIndexableGetter<MemorySpace, Geometry>;

  return ArborX::DistributedTree<MemorySpace, PairIndexRank, IndexableGetter>(
      comm, ExecutionSpace{}, pairs, IndexableGetter{geometries});
}
#endif

template <typename DeviceType, typename Geometry>
auto makeIntersectsQueries(
    std::vector<Geometry> const &geometries,
    typename DeviceType::execution_space const &exec_space = {})
{
  int const n = geometries.size();
  Kokkos::View<decltype(ArborX::intersects(Geometry{})) *, DeviceType> queries(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "Testing::intersecting_with_box_predicates"),
      n);
  auto queries_host =
      Kokkos::create_mirror_view(Kokkos::WithoutInitializing, queries);
  for (int i = 0; i < n; ++i)
    queries_host(i) = ArborX::intersects(geometries[i]);
  Kokkos::deep_copy(exec_space, queries, queries_host);
  return queries;
}

template <typename DeviceType, typename Geometry, typename Data>
auto makeIntersectsWithAttachmentQueries(
    std::vector<Geometry> const &geometries, std::vector<Data> const &data,
    typename DeviceType::execution_space const &exec_space = {})
{
  int const n = geometries.size();
  Kokkos::View<decltype(ArborX::attach(ArborX::intersects(Geometry{}),
                                       Data{})) *,
               DeviceType>
      queries(Kokkos::view_alloc(
                  Kokkos::WithoutInitializing,
                  "Testing::intersecting_with_box_with_attachment_predicates"),
              n);
  auto queries_host =
      Kokkos::create_mirror_view(Kokkos::WithoutInitializing, queries);
  for (int i = 0; i < n; ++i)
    queries_host(i) =
        ArborX::attach(ArborX::intersects(geometries[i]), data[i]);
  Kokkos::deep_copy(exec_space, queries, queries_host);
  return queries;
}

template <typename DeviceType, typename Geometry>
auto makeNearestQueries(
    std::vector<std::pair<Geometry, int>> const &geometries,
    typename DeviceType::execution_space const &exec_space = {})
{
  // NOTE: the individial geometry elements in the input are pairs of an actual
  // point and a number k of neighbors to query for.
  int const n = geometries.size();
  Kokkos::View<ArborX::Nearest<Geometry> *, DeviceType> queries(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "Testing::nearest_queries"),
      n);
  auto queries_host =
      Kokkos::create_mirror_view(Kokkos::WithoutInitializing, queries);
  for (int i = 0; i < n; ++i)
    queries_host(i) =
        ArborX::nearest(geometries[i].first, geometries[i].second);
  Kokkos::deep_copy(exec_space, queries, queries_host);
  return queries;
}

template <typename DeviceType, typename Geometry, typename Data>
auto makeNearestWithAttachmentQueries(
    std::vector<std::pair<Geometry, int>> const &points,
    std::vector<Data> const &data,
    typename DeviceType::execution_space const &exec_space = {})
{
  // NOTE: `points` is not a very descriptive name here. It stores both the
  // actual point and the number k of neighbors to query for.
  int const n = points.size();
  Kokkos::View<decltype(ArborX::attach(ArborX::Nearest<Geometry>{}, Data{})) *,
               DeviceType>
      queries(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                 "Testing::nearest_queries"),
              n);
  auto queries_host =
      Kokkos::create_mirror_view(Kokkos::WithoutInitializing, queries);
  for (int i = 0; i < n; ++i)
    queries_host(i) = ArborX::attach(
        ArborX::nearest(points[i].first, points[i].second), data[i]);
  Kokkos::deep_copy(exec_space, queries, queries_host);
  return queries;
}

#endif
