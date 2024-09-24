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
#include "ArborX_PairIndexRank.hpp"
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
template <typename D>
struct is_distributed<ArborX::DistributedTree<D>> : std::true_type
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
#ifdef ARBORX_ENABLE_MPI
  using value_type =
      std::conditional_t<is_distributed<Tree>{}, ArborX::PairIndexRank, int>;
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

template <typename Tree, typename ExecutionSpace, int DIM = 3,
          typename Coordinate = float>
auto make(ExecutionSpace const &exec_space,
          std::vector<ArborX::Box<DIM, Coordinate>> const &b)
{
  int const n = b.size();
  Kokkos::View<ArborX::Box<DIM, Coordinate> *, typename Tree::memory_space>
      boxes(Kokkos::view_alloc(Kokkos::WithoutInitializing, "Testing::boxes"),
            n);
  auto boxes_host =
      Kokkos::create_mirror_view(Kokkos::WithoutInitializing, boxes);
  for (int i = 0; i < n; ++i)
    boxes_host(i) = b[i];
  Kokkos::deep_copy(exec_space, boxes, boxes_host);
  return Tree(exec_space, boxes);
}

#ifdef ARBORX_ENABLE_MPI
template <typename DeviceType, int DIM = 3, typename Coordinate = float>
auto makeDistributedTree(MPI_Comm comm,
                         std::vector<ArborX::Box<DIM, Coordinate>> const &b)
{
  using ExecutionSpace = typename DeviceType::execution_space;

  int const n = b.size();
  Kokkos::View<ArborX::Box<DIM, Coordinate> *, DeviceType> boxes(
      "Testing::boxes", n);
  auto boxes_host = Kokkos::create_mirror_view(boxes);
  for (int i = 0; i < n; ++i)
    boxes_host(i) = b[i];
  Kokkos::deep_copy(boxes, boxes_host);

  return ArborX::DistributedTree<typename DeviceType::memory_space>(
      comm, ExecutionSpace{}, boxes);
}
#endif

template <typename DeviceType, int DIM = 3, typename Coordinate = float>
auto makeIntersectsBoxQueries(
    std::vector<ArborX::Box<DIM, Coordinate>> const &boxes,
    typename DeviceType::execution_space const &exec_space = {})
{
  int const n = boxes.size();
  Kokkos::View<decltype(ArborX::intersects(ArborX::Box<DIM, Coordinate>{})) *,
               DeviceType>
      queries(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                 "Testing::intersecting_with_box_predicates"),
              n);
  auto queries_host =
      Kokkos::create_mirror_view(Kokkos::WithoutInitializing, queries);
  for (int i = 0; i < n; ++i)
    queries_host(i) = ArborX::intersects(boxes[i]);
  Kokkos::deep_copy(exec_space, queries, queries_host);
  return queries;
}

template <typename DeviceType, typename Data, int DIM = 3,
          typename Coordinate = float>
auto makeIntersectsBoxWithAttachmentQueries(
    std::vector<ArborX::Box<DIM, Coordinate>> const &boxes,
    std::vector<Data> const &data,
    typename DeviceType::execution_space const &exec_space = {})
{
  int const n = boxes.size();
  Kokkos::View<decltype(ArborX::attach(
                   ArborX::intersects(ArborX::Box<DIM, Coordinate>{}),
                   Data{})) *,
               DeviceType>
      queries(Kokkos::view_alloc(
                  Kokkos::WithoutInitializing,
                  "Testing::intersecting_with_box_with_attachment_predicates"),
              n);
  auto queries_host =
      Kokkos::create_mirror_view(Kokkos::WithoutInitializing, queries);
  for (int i = 0; i < n; ++i)
    queries_host(i) = ArborX::attach(ArborX::intersects(boxes[i]), data[i]);
  Kokkos::deep_copy(exec_space, queries, queries_host);
  return queries;
}

template <typename DeviceType, int DIM = 3, typename Coordinate = float>
auto makeNearestQueries(
    std::vector<std::pair<ArborX::Point<DIM, Coordinate>, int>> const &points,
    typename DeviceType::execution_space const &exec_space = {})
{
  // NOTE: `points` is not a very descriptive name here. It stores both the
  // actual point and the number k of neighbors to query for.
  int const n = points.size();
  Kokkos::View<ArborX::Nearest<ArborX::Point<DIM, Coordinate>> *, DeviceType>
      queries(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                 "Testing::nearest_queries"),
              n);
  auto queries_host =
      Kokkos::create_mirror_view(Kokkos::WithoutInitializing, queries);
  for (int i = 0; i < n; ++i)
    queries_host(i) = ArborX::nearest(points[i].first, points[i].second);
  Kokkos::deep_copy(exec_space, queries, queries_host);
  return queries;
}

template <typename DeviceType, int DIM = 3, typename Coordinate = float>
auto makeBoxNearestQueries(
    std::vector<std::tuple<ArborX::Point<DIM, Coordinate>,
                           ArborX::Point<DIM, Coordinate>, int>> const &boxes,
    typename DeviceType::execution_space const &exec_space = {})
{
  // NOTE: `boxes` is not a very descriptive name here. It stores both the
  // corners of the boxe and the number k of neighbors to query for.
  using Box = ArborX::Box<DIM, Coordinate>;
  int const n = boxes.size();
  Kokkos::View<ArborX::Nearest<Box> *, DeviceType> queries(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "Testing::nearest_queries"),
      n);
  auto queries_host =
      Kokkos::create_mirror_view(Kokkos::WithoutInitializing, queries);
  for (int i = 0; i < n; ++i)
    queries_host(i) =
        ArborX::nearest(Box{std::get<0>(boxes[i]), std::get<1>(boxes[i])},
                        std::get<2>(boxes[i]));
  Kokkos::deep_copy(exec_space, queries, queries_host);
  return queries;
}

template <typename DeviceType, int DIM = 3, typename Coordinate = float>
auto makeSphereNearestQueries(
    std::vector<std::tuple<ArborX::Point<DIM, Coordinate>, float, int>> const
        &spheres,
    typename DeviceType::execution_space const &exec_space = {})
{
  // NOTE: `sphere` is not a very descriptive name here. It stores both the
  // center and the radius of the sphere and the number k of neighbors to query
  // for.
  using Sphere = ArborX::Sphere<DIM, Coordinate>;
  int const n = spheres.size();
  Kokkos::View<ArborX::Nearest<Sphere> *, DeviceType> queries(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "Testing::nearest_queries"),
      n);
  auto queries_host =
      Kokkos::create_mirror_view(Kokkos::WithoutInitializing, queries);
  for (int i = 0; i < n; ++i)
    queries_host(i) = ArborX::nearest(
        Sphere{std::get<0>(spheres[i]), std::get<1>(spheres[i])},
        std::get<2>(spheres[i]));
  Kokkos::deep_copy(exec_space, queries, queries_host);
  return queries;
}

template <typename DeviceType, typename Data, int DIM = 3,
          typename Coordinate = float>
auto makeNearestWithAttachmentQueries(
    std::vector<std::pair<ArborX::Point<DIM, Coordinate>, int>> const &points,
    std::vector<Data> const &data,
    typename DeviceType::execution_space const &exec_space = {})
{
  // NOTE: `points` is not a very descriptive name here. It stores both the
  // actual point and the number k of neighbors to query for.
  int const n = points.size();
  Kokkos::View<decltype(ArborX::attach(
                   ArborX::Nearest<ArborX::Point<DIM, Coordinate>>{},
                   Data{})) *,
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

template <typename DeviceType, int DIM = 3, typename Coordinate = float>
auto makeIntersectsSphereQueries(
    std::vector<std::pair<ArborX::Point<DIM, Coordinate>, float>> const &points,
    typename DeviceType::execution_space const &exec_space = {})
{
  // NOTE: `points` is not a very descriptive name here. It stores both the
  // actual point and the radius for the search around that point.
  using Sphere = ArborX::Sphere<DIM, Coordinate>;
  int const n = points.size();
  Kokkos::View<decltype(ArborX::intersects(Sphere{})) *, DeviceType> queries(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "Testing::intersecting_with_sphere_predicates"),
      n);
  auto queries_host =
      Kokkos::create_mirror_view(Kokkos::WithoutInitializing, queries);
  for (int i = 0; i < n; ++i)
    queries_host(i) =
        ArborX::intersects(Sphere{points[i].first, points[i].second});
  Kokkos::deep_copy(exec_space, queries, queries_host);
  return queries;
}

#endif
