/****************************************************************************
 * Copyright (c) 2012-2019 by the ArborX authors                            *
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
#include "boost_ext/TupleComparison.hpp"
#include "CompressedSparseRow.hpp"
#include "VectorOfTuples.hpp"
// clang-format on

#include "ArborX_EnableViewComparison.hpp"
#include <ArborX_DetailsKokkosExt.hpp> // is_accessible_from
#ifdef ARBORX_ENABLE_MPI
#include <ArborX_DistributedSearchTree.hpp>
#endif
#include <ArborX_LinearBVH.hpp>

#include <Kokkos_View.hpp>

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

template <typename T1, typename T2>
void validateResults(T1 const &reference, T2 const &other)
{
  auto const m = getNumberOfRows(reference);
  BOOST_TEST(m == getNumberOfRows(other));
  for (std::size_t i = 0; i < m; ++i)
  {
    auto const l = extractRow(other, i);
    auto const r = extractRow(reference, i);
    BOOST_TEST(l == r, boost::test_tools::per_element());
  }
}

namespace tt = boost::test_tools;

template <typename Query, typename DeviceType>
void checkResults(ArborX::BVH<DeviceType> const &bvh,
                  Kokkos::View<Query *, DeviceType> const &queries,
                  std::vector<int> const &indices_ref,
                  std::vector<int> const &offset_ref)
{
  Kokkos::View<int *, DeviceType> indices("indices", 0);
  Kokkos::View<int *, DeviceType> offset("offset", 0);
  bvh.query(queries, indices, offset);

  auto indices_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, indices);
  auto offset_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offset);

  validateResults(std::make_tuple(offset_host, indices_host),
                  std::make_tuple(offset_ref, indices_ref));
}

// Same as above except that we get the distances out of the queries and
// compare them to the reference solution passed as argument.  Templated type
// `Query` is pretty much a nearest predicate in this case.
template <typename Query, typename DeviceType>
void checkResults(ArborX::BVH<DeviceType> const &bvh,
                  Kokkos::View<Query *, DeviceType> const &queries,
                  std::vector<int> const &indices_ref,
                  std::vector<int> const &offset_ref,
                  std::vector<double> const &distances_ref)
{
  Kokkos::View<int *, DeviceType> indices("indices", 0);
  Kokkos::View<int *, DeviceType> offset("offset", 0);
  Kokkos::View<double *, DeviceType> distances("distances", 0);
  bvh.query(queries, indices, offset, distances);

  auto indices_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, indices);
  auto offset_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offset);
  auto distances_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, distances);

  validateResults(std::make_tuple(offset_host, indices_host, distances_host),
                  std::make_tuple(offset_ref, indices_ref, distances_ref));
}

#ifdef ARBORX_ENABLE_MPI
template <typename Query, typename DeviceType>
void checkResults(ArborX::DistributedSearchTree<DeviceType> const &tree,
                  Kokkos::View<Query *, DeviceType> const &queries,
                  std::vector<int> const &indices_ref,
                  std::vector<int> const &offset_ref,
                  std::vector<int> const &ranks_ref)
{
  Kokkos::View<int *, DeviceType> indices("indices", 0);
  Kokkos::View<int *, DeviceType> offset("offset", 0);
  Kokkos::View<int *, DeviceType> ranks("ranks", 0);
  tree.query(queries, indices, offset, ranks);

  auto indices_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, indices);
  auto offset_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offset);
  auto ranks_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, ranks);

  validateResults(std::make_tuple(offset_host, ranks_host, indices_host),
                  std::make_tuple(offset_ref, ranks_ref, indices_ref));
}

template <typename Query, typename DeviceType>
void checkResults(ArborX::DistributedSearchTree<DeviceType> const &tree,
                  Kokkos::View<Query *, DeviceType> const &queries,
                  std::vector<int> const &indices_ref,
                  std::vector<int> const &offset_ref,
                  std::vector<int> const &ranks_ref,
                  std::vector<double> const &distances_ref)
{
  Kokkos::View<int *, DeviceType> indices("indices", 0);
  Kokkos::View<int *, DeviceType> offset("offset", 0);
  Kokkos::View<int *, DeviceType> ranks("ranks", 0);
  Kokkos::View<double *, DeviceType> distances("distances", 0);
  tree.query(queries, indices, offset, ranks, distances);

  auto indices_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, indices);
  auto offset_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offset);
  auto ranks_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, ranks);
  auto distances_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, distances);

  validateResults(
      std::make_tuple(offset_host, ranks_host, indices_host, distances_host),
      std::make_tuple(offset_ref, ranks_ref, indices_ref, distances_ref));
}
#endif

template <typename DeviceType>
ArborX::BVH<DeviceType> makeBvh(std::vector<ArborX::Box> const &b)
{
  int const n = b.size();
  Kokkos::View<ArborX::Box *, DeviceType> boxes("boxes", n);
  auto boxes_host = Kokkos::create_mirror_view(boxes);
  for (int i = 0; i < n; ++i)
    boxes_host(i) = b[i];
  Kokkos::deep_copy(boxes, boxes_host);
  return ArborX::BVH<DeviceType>(boxes);
}

#ifdef ARBORX_ENABLE_MPI
template <typename DeviceType>
ArborX::DistributedSearchTree<DeviceType>
makeDistributedSearchTree(MPI_Comm comm, std::vector<ArborX::Box> const &b)
{
  int const n = b.size();
  Kokkos::View<ArborX::Box *, DeviceType> boxes("boxes", n);
  auto boxes_host = Kokkos::create_mirror_view(boxes);
  for (int i = 0; i < n; ++i)
    boxes_host(i) = b[i];
  Kokkos::deep_copy(boxes, boxes_host);
  return ArborX::DistributedSearchTree<DeviceType>(comm, boxes);
}
#endif

template <typename DeviceType>
Kokkos::View<ArborX::Overlap *, DeviceType>
makeOverlapQueries(std::vector<ArborX::Box> const &boxes)
{
  int const n = boxes.size();
  Kokkos::View<ArborX::Overlap *, DeviceType> queries("overlap_queries", n);
  auto queries_host = Kokkos::create_mirror_view(queries);
  for (int i = 0; i < n; ++i)
    queries_host(i) = ArborX::overlap(boxes[i]);
  Kokkos::deep_copy(queries, queries_host);
  return queries;
}

template <typename DeviceType>
Kokkos::View<ArborX::Nearest<ArborX::Point> *, DeviceType>
makeNearestQueries(std::vector<std::pair<ArborX::Point, int>> const &points)
{
  // NOTE: `points` is not a very descriptive name here. It stores both the
  // actual point and the number k of neighbors to query for.
  int const n = points.size();
  Kokkos::View<ArborX::Nearest<ArborX::Point> *, DeviceType> queries(
      "nearest_queries", n);
  auto queries_host = Kokkos::create_mirror_view(queries);
  for (int i = 0; i < n; ++i)
    queries_host(i) = ArborX::nearest(points[i].first, points[i].second);
  Kokkos::deep_copy(queries, queries_host);
  return queries;
}

template <typename DeviceType>
Kokkos::View<ArborX::Within *, DeviceType>
makeWithinQueries(std::vector<std::pair<ArborX::Point, double>> const &points)
{
  // NOTE: `points` is not a very descriptive name here. It stores both the
  // actual point and the radius for the search around that point.
  int const n = points.size();
  Kokkos::View<ArborX::Within *, DeviceType> queries("within_queries", n);
  auto queries_host = Kokkos::create_mirror_view(queries);
  for (int i = 0; i < n; ++i)
    queries_host(i) = ArborX::within(points[i].first, points[i].second);
  Kokkos::deep_copy(queries, queries_host);
  return queries;
}

#endif
