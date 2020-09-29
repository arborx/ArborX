/****************************************************************************
 * Copyright (c) 2012-2020 by the ArborX authors                            *
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
#include "VectorOfTuples.hpp"
// clang-format on

#include "ArborX_BoostRTreeHelpers.hpp"
#include "ArborX_EnableViewComparison.hpp"
#include <ArborX_DetailsKokkosExt.hpp> // is_accessible_from
#ifdef ARBORX_ENABLE_MPI
#include <ArborX_DistributedTree.hpp>
#endif
#include <ArborX_LinearBVH.hpp>

#include <boost/test/unit_test.hpp>

#include <iostream>
#include <tuple>
#include <vector>

namespace Details
{

template <typename... Ps>
struct ArrayTraits<Kokkos::View<Ps...>>
{

  using array_type = Kokkos::View<Ps...>;
  static_assert(array_type::rank == 1, "requires rank-1 views");
  using value_type = typename array_type::value_type;
  static std::size_t size(array_type const &v) { return v.extent(0); }
  static value_type const &access(array_type const &v, std::size_t i)
  {
    return v(i);
  }
};

template <typename T>
struct ArrayTraits<std::vector<T>>
{
  using array_type = std::vector<T>;
  using value_type = typename array_type::value_type;
  static std::size_t size(array_type const &v) { return v.size(); }
  static value_type const &access(array_type const &v, std::size_t i)
  {
    return v[i];
  }
};

} // namespace Details

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

template <typename Tree, typename Queries>
auto query(Tree const &tree, Queries const &queries)
{
  using device_type = typename Tree::device_type;
  using value_type =
      std::conditional_t<is_distributed<Tree>{}, Kokkos::pair<int, int>, int>;
  Kokkos::View<value_type *, device_type> values("Testing::values", 0);
  Kokkos::View<int *, device_type> offsets("Testing::offsets", 0);
  tree.query(queries, values, offsets);
  return make_compressed_storage(
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offsets),
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, values));
}

#define ARBORX_TEST_QUERY_TREE(tree, queries, reference)                       \
  BOOST_TEST(query(tree, queries) == (reference),                              \
             boost::test_tools::per_element());

template <typename Tree, typename Queries>
auto query_with_distance(Tree const &tree, Queries const &queries,
                         std::enable_if_t<!is_distributed<Tree>{}> * = nullptr)
{
  using device_type = typename Tree::device_type;
  Kokkos::View<Kokkos::pair<int, float> *, device_type> values(
      "Testing::values", 0);
  Kokkos::View<int *, device_type> offsets("Testing::offsets", 0);
  tree.query(queries,
             ArborX::Details::CallbackDefaultNearestPredicateWithDistance{},
             values, offsets);
  return make_compressed_storage(
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offsets),
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, values));
}

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
template <typename Tree, typename Queries>
auto query_with_distance(Tree const &tree, Queries const &queries,
                         std::enable_if_t<is_distributed<Tree>{}> * = nullptr)
{
  using device_type = typename Tree::device_type;
  Kokkos::View<int *, device_type> offsets("Testing::offsets", 0);
  Kokkos::View<int *, device_type> indices("Testing::indices", 0);
  Kokkos::View<int *, device_type> ranks("Testing::ranks", 0);
  Kokkos::View<float *, device_type> distances("Testing::distances", 0);
  using ExecutionSpace = typename device_type::execution_space;
  ExecutionSpace space;
  ArborX::Details::DistributedTreeImpl<device_type>::queryDispatchImpl(
      ArborX::Details::NearestPredicateTag{}, tree, space, queries, indices,
      offsets, ranks, &distances);

  auto values = zip(space, indices, ranks, distances);

  return make_compressed_storage(
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offsets),
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, values));
}
#endif

#define ARBORX_TEST_QUERY_TREE_WITH_DISTANCE(tree, queries, reference)         \
  BOOST_TEST(query_with_distance(tree, queries) == (reference),                \
             boost::test_tools::per_element());

template <typename Tree>
auto make(std::vector<ArborX::Box> const &b)
{
  int const n = b.size();
  Kokkos::View<ArborX::Box *, typename Tree::device_type> boxes(
      "Testing::boxes", n);
  auto boxes_host = Kokkos::create_mirror_view(boxes);
  for (int i = 0; i < n; ++i)
    boxes_host(i) = b[i];
  Kokkos::deep_copy(boxes, boxes_host);
  return Tree(boxes);
}

#ifdef ARBORX_ENABLE_MPI
template <typename DeviceType>
ArborX::DistributedTree<DeviceType>
makeDistributedTree(MPI_Comm comm, std::vector<ArborX::Box> const &b)
{
  int const n = b.size();
  Kokkos::View<ArborX::Box *, DeviceType> boxes("Testing::boxes", n);
  auto boxes_host = Kokkos::create_mirror_view(boxes);
  for (int i = 0; i < n; ++i)
    boxes_host(i) = b[i];
  Kokkos::deep_copy(boxes, boxes_host);
  return ArborX::DistributedTree<DeviceType>(comm, boxes);
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
