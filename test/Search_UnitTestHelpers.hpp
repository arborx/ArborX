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

#ifndef ARBORX_SEARCH_TEST_HELPERS_HPP
#define ARBORX_SEARCH_TEST_HELPERS_HPP

// clang-format off
#include "boost_ext/KokkosPairComparison.hpp"
#include "boost_ext/TupleComparison.hpp"
#include "boost_ext/CompressedStorageComparison.hpp"
// clang-format on

#include "ArborX_EnableViewComparison.hpp"
#ifdef ARBORX_ENABLE_MPI
#include "ArborX_BoostRTreeHelpers.hpp"
#include <ArborX_DistributedTree.hpp>
#endif

#include <boost/test/unit_test.hpp>

#include <iostream>
#include <utility>
#include <vector>

template <typename T>
struct is_distributed : std::false_type
{
};

#ifdef ARBORX_ENABLE_MPI
template <typename D>
struct is_distributed<ArborX::DistributedTree<D>> : std::true_type
{
};

template <typename I>
struct is_distributed<BoostExt::ParallelRTree<I>> : std::true_type
{
};
#endif

template <typename T>
auto make_reference_solution(std::vector<T> const &values,
                             std::vector<int> const &offsets)
{
  return make_compressed_storage(offsets, values);
}

#ifdef ARBORX_ENABLE_MPI
// FIXME This is a temporary workaround until we reconcile interfaces of
// DistributedTree and BVH
template <typename ExecutionSpace, typename MemorySpace, typename Queries,
          typename Values, typename Offsets>
void query(ArborX::DistributedTree<MemorySpace> const &tree,
           ExecutionSpace const &space, Queries const &queries,
           Values const &values, Offsets const &offsets)
{
  tree.query(space, queries, values, offsets);
}
#endif

template <typename ExecutionSpace, typename Tree, typename Queries>
auto query(ExecutionSpace const &exec_space, Tree const &tree,
           Queries const &queries)
{
  using memory_space = typename Tree::memory_space;
  using value_type =
      std::conditional_t<is_distributed<Tree>{}, Kokkos::pair<int, int>, int>;
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

// Workaround for NVCC that complains that the enclosing parent function
// (query_with_distance) for an extended __host__ __device__ lambda must not
// have deduced return type
template <typename DeviceType, typename ExecutionSpace>
Kokkos::View<Kokkos::pair<Kokkos::pair<int, int>, float> *, DeviceType>
zip(ExecutionSpace const &space, Kokkos::View<int *, DeviceType> indices,
    Kokkos::View<int *, DeviceType> ranks,
    Kokkos::View<float *, DeviceType> distances)
{
  auto const n = indices.extent(0);
  Kokkos::View<Kokkos::pair<Kokkos::pair<int, int>, float> *, DeviceType>
      values(Kokkos::view_alloc(Kokkos::WithoutInitializing, "Testing::values"),
             n);
  Kokkos::parallel_for("ArborX:UnitTestSupport:zip",
                       Kokkos::RangePolicy<ExecutionSpace>(space, 0, n),
                       KOKKOS_LAMBDA(int i) {
                         values(i) = {{indices(i), ranks(i)}, distances(i)};
                       });
  return values;
}

#ifdef ARBORX_ENABLE_MPI
template <typename ExecutionSpace, typename Tree, typename Queries>
auto query_with_distance(ExecutionSpace const &exec_space, Tree const &tree,
                         Queries const &queries,
                         std::enable_if_t<is_distributed<Tree>{}> * = nullptr)
{
  using MemorySpace = typename Tree::memory_space;
  using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;

  Kokkos::View<int *, MemorySpace> offsets("Testing::offsets", 0);
  Kokkos::View<int *, MemorySpace> indices("Testing::indices", 0);
  Kokkos::View<int *, MemorySpace> ranks("Testing::ranks", 0);
  Kokkos::View<float *, MemorySpace> distances("Testing::distances", 0);
  ArborX::Details::DistributedTreeImpl<DeviceType>::queryDispatchImpl(
      ArborX::Details::NearestPredicateTag{}, tree, exec_space, queries,
      indices, offsets, ranks, &distances);

  auto values = zip(exec_space, indices, ranks, distances);

  return make_compressed_storage(
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offsets),
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, values));
}
#endif

#define ARBORX_TEST_QUERY_TREE_WITH_DISTANCE(exec_space, tree, queries,        \
                                             reference)                        \
  BOOST_TEST(query_with_distance(exec_space, tree, queries) == (reference),    \
             boost::test_tools::per_element());

template <typename Tree, typename ExecutionSpace>
auto make(ExecutionSpace const &exec_space, std::vector<ArborX::Box> const &b)
{
  int const n = b.size();
  Kokkos::View<ArborX::Box *, typename Tree::memory_space> boxes(
      "Testing::boxes", n);
  auto boxes_host = Kokkos::create_mirror_view(boxes);
  for (int i = 0; i < n; ++i)
    boxes_host(i) = b[i];
  Kokkos::deep_copy(boxes, boxes_host);
  return Tree(exec_space, boxes);
}

#ifdef ARBORX_ENABLE_MPI
template <typename DeviceType>
ArborX::DistributedTree<typename DeviceType::memory_space>
makeDistributedTree(MPI_Comm comm, std::vector<ArborX::Box> const &b)
{
  using ExecutionSpace = typename DeviceType::execution_space;

  int const n = b.size();
  Kokkos::View<ArborX::Box *, DeviceType> boxes("Testing::boxes", n);
  auto boxes_host = Kokkos::create_mirror_view(boxes);
  for (int i = 0; i < n; ++i)
    boxes_host(i) = b[i];
  Kokkos::deep_copy(boxes, boxes_host);

  return ArborX::DistributedTree<typename DeviceType::memory_space>(
      comm, ExecutionSpace{}, boxes);
}
#endif

template <typename DeviceType>
auto makeIntersectsBoxQueries(std::vector<ArborX::Box> const &boxes)
{
  int const n = boxes.size();
  Kokkos::View<decltype(ArborX::intersects(ArborX::Box{})) *, DeviceType>
      queries("Testing::intersecting_with_box_predicates", n);
  auto queries_host = Kokkos::create_mirror_view(queries);
  for (int i = 0; i < n; ++i)
    queries_host(i) = ArborX::intersects(boxes[i]);
  Kokkos::deep_copy(queries, queries_host);
  return queries;
}

template <typename DeviceType, typename Data>
auto makeIntersectsBoxWithAttachmentQueries(
    std::vector<ArborX::Box> const &boxes, std::vector<Data> const &data)
{
  int const n = boxes.size();
  Kokkos::View<decltype(
                   ArborX::attach(ArborX::intersects(ArborX::Box{}), Data{})) *,
               DeviceType>
      queries("Testing::intersecting_with_box_with_attachment_predicates", n);
  auto queries_host = Kokkos::create_mirror_view(queries);
  for (int i = 0; i < n; ++i)
    queries_host(i) = ArborX::attach(ArborX::intersects(boxes[i]), data[i]);
  Kokkos::deep_copy(queries, queries_host);
  return queries;
}

template <typename DeviceType>
auto makeNearestQueries(
    std::vector<std::pair<ArborX::Point, int>> const &points)
{
  // NOTE: `points` is not a very descriptive name here. It stores both the
  // actual point and the number k of neighbors to query for.
  int const n = points.size();
  Kokkos::View<ArborX::Nearest<ArborX::Point> *, DeviceType> queries(
      "Testing::nearest_queries", n);
  auto queries_host = Kokkos::create_mirror_view(queries);
  for (int i = 0; i < n; ++i)
    queries_host(i) = ArborX::nearest(points[i].first, points[i].second);
  Kokkos::deep_copy(queries, queries_host);
  return queries;
}

template <typename DeviceType, typename Data>
auto makeNearestWithAttachmentQueries(
    std::vector<std::pair<ArborX::Point, int>> const &points,
    std::vector<Data> const &data)
{
  // NOTE: `points` is not a very descriptive name here. It stores both the
  // actual point and the number k of neighbors to query for.
  int const n = points.size();
  Kokkos::View<decltype(
                   ArborX::attach(ArborX::Nearest<ArborX::Point>{}, Data{})) *,
               DeviceType>
      queries("Testing::nearest_queries", n);
  auto queries_host = Kokkos::create_mirror_view(queries);
  for (int i = 0; i < n; ++i)
    queries_host(i) = ArborX::attach(
        ArborX::nearest(points[i].first, points[i].second), data[i]);
  Kokkos::deep_copy(queries, queries_host);
  return queries;
}

template <typename DeviceType>
Kokkos::View<decltype(ArborX::intersects(ArborX::Sphere{})) *, DeviceType>
makeIntersectsSphereQueries(
    std::vector<std::pair<ArborX::Point, float>> const &points)
{
  // NOTE: `points` is not a very descriptive name here. It stores both the
  // actual point and the radius for the search around that point.
  int const n = points.size();
  Kokkos::View<decltype(ArborX::intersects(ArborX::Sphere{})) *, DeviceType>
      queries("Testing::intersecting_with_sphere_predicates", n);
  auto queries_host = Kokkos::create_mirror_view(queries);
  for (int i = 0; i < n; ++i)
    queries_host(i) =
        ArborX::intersects(ArborX::Sphere{points[i].first, points[i].second});
  Kokkos::deep_copy(queries, queries_host);
  return queries;
}

#endif
