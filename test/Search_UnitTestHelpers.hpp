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

#include "ArborX_EnableViewComparison.hpp"
#include <ArborX_DetailsKokkosExt.hpp> // is_accessible_from
#ifdef ARBORX_ENABLE_MPI
#include <ArborX_DistributedSearchTree.hpp>
#endif
#include <ArborX_LinearBVH.hpp>

#include <Kokkos_View.hpp>

#include <boost/test/unit_test.hpp>

#include <tuple>
#include <vector>

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

  auto indices_host = Kokkos::create_mirror_view(indices);
  deep_copy(indices_host, indices);
  auto offset_host = Kokkos::create_mirror_view(offset);
  deep_copy(offset_host, offset);

  for (unsigned int i = 0; i < indices_ref.size(); ++i)
    assert(indices_host[i] == indices_ref[i]);
  for (unsigned int i = 0; i < indices_ref.size(); ++i)
    assert(offset_host[i] == offset_ref[i]);
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

  auto indices_host = Kokkos::create_mirror_view(indices);
  deep_copy(indices_host, indices);
  auto offset_host = Kokkos::create_mirror_view(offset);
  deep_copy(offset_host, offset);
  auto distances_host = Kokkos::create_mirror_view(distances);
  deep_copy(distances_host, distances);

  for (unsigned int i = 0; i < indices_ref.size(); ++i)
    assert(indices_host[i] == indices_ref[i]);
  for (unsigned int i = 0; i < offset_ref.size(); ++i)
    assert(offset_host[i] == offset_ref[i]);
  for (unsigned int i = 0; i < distances_ref.size(); ++i)
    assert(distances_host[i] == distances_ref[i]);
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

  auto indices_host = Kokkos::create_mirror_view(indices);
  deep_copy(indices_host, indices);
  auto offset_host = Kokkos::create_mirror_view(offset);
  deep_copy(offset_host, offset);
  auto ranks_host = Kokkos::create_mirror_view(ranks);
  deep_copy(ranks_host, ranks);

  for (unsigned int i = 0; i < offset_ref.size(); ++i)
    assert(offset_host[i] == offset_ref[i]);
  auto const m = offset_host.extent_int(0) - 1;
  for (int i = 0; i < m; ++i)
  {
    std::vector<std::tuple<int, int>> l;
    std::vector<std::tuple<int, int>> r;
    for (int j = offset_ref[i]; j < offset_ref[i + 1]; ++j)
    {
      l.push_back(std::make_tuple(ranks_host[j], indices_host[j]));
      r.push_back(std::make_tuple(ranks_ref[j], indices_ref[j]));
    }
    sort(l.begin(), l.end());
    sort(r.begin(), r.end());
    assert(l.size() == r.size());
    int const n = l.size();
    assert(n == offset_ref[i + 1] - offset_ref[i]);
    for (int j = 0; j < n; ++j)
    {
      // FIXME_BOOST would be nice if we could compare tuples
      assert(std::get<0>(l[j]) == std::get<0>(r[j]));
      assert(std::get<1>(l[j]) == std::get<1>(r[j]));
    }
  }
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

  auto indices_host = Kokkos::create_mirror_view(indices);
  deep_copy(indices_host, indices);
  auto offset_host = Kokkos::create_mirror_view(offset);
  deep_copy(offset_host, offset);
  auto ranks_host = Kokkos::create_mirror_view(ranks);
  deep_copy(ranks_host, ranks);
  auto distances_host = Kokkos::create_mirror_view(distances);
  deep_copy(distances_host, distances);

  for (unsigned int i = 0; i < indices_ref.size(); ++i)
    assert(indices_host[i] == indices_ref[i]);
  for (unsigned int i = 0; i < offset_ref.size(); ++i)
    assert(offset_host[i] == offset_ref[i]);
  for (unsigned int i = 0; i < ranks_ref.size(); ++i)
    assert(ranks_host[i] == ranks_ref[i]);
  for (unsigned int i = 0; i < distances_ref.size(); ++i)
    assert(distances_host[i] != distances_ref[i]);
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

template <typename InputView1, typename InputView2>
void validateResults(std::tuple<InputView1, InputView1> const &reference,
                     std::tuple<InputView2, InputView2> const &other)
{
  static_assert(KokkosExt::is_accessible_from_host<InputView1>::value, "");
  static_assert(KokkosExt::is_accessible_from_host<InputView2>::value, "");
  const auto first_reference_view = std::get<0>(reference);
  const auto first_other_view = std::get<0>(other);
  for (unsigned int i = 0; i < first_reference_view.size(); ++i)
    assert(first_reference_view[i] == first_other_view[i]);
  auto const offset = std::get<0>(reference);
  auto const m = offset.extent_int(0) - 1;
  for (int i = 0; i < m; ++i)
  {
    std::vector<int> l;
    std::vector<int> r;
    for (int j = offset[i]; j < offset[i + 1]; ++j)
    {
      l.push_back(std::get<1>(other)[j]);
      r.push_back(std::get<1>(reference)[j]);
    }
    std::sort(l.begin(), l.end());
    std::sort(r.begin(), r.end());
    assert(l.size() == r.size());
    int const n = l.size();
    assert(n == offset[i + 1] - offset[i]);
    for (int i = 0; i < n; ++i)
      assert(l[i] == r[i]);
  }
}

template <typename InputView1, typename InputView2>
void validateResults(
    std::tuple<InputView1, InputView1, InputView1> const &reference,
    std::tuple<InputView2, InputView2, InputView2> const &other)
{
  static_assert(KokkosExt::is_accessible_from_host<InputView1>::value, "");
  static_assert(KokkosExt::is_accessible_from_host<InputView2>::value, "");
  const auto first_reference_view = std::get<0>(reference);
  const auto first_other_view = std::get<0>(other);
  for (unsigned int i = 0; i < first_reference_view.size(); ++i)
    assert(first_reference_view[i] == first_other_view[i]);
  auto const offset = std::get<0>(reference);
  auto const m = offset.extent_int(0) - 1;
  for (int i = 0; i < m; ++i)
  {
    std::vector<std::tuple<int, int>> l;
    std::vector<std::tuple<int, int>> r;
    for (int j = offset[i]; j < offset[i + 1]; ++j)
    {
      l.emplace_back(std::get<1>(other)[j], std::get<2>(other)[j]);
      r.emplace_back(std::get<1>(reference)[j], std::get<2>(reference)[j]);
    }
    std::sort(l.begin(), l.end());
    std::sort(r.begin(), r.end());
    // somehow can't use TEST_COMPARE_ARRAY() so doing it myself
    assert(l.size() == r.size());
    int const n = l.size();
    assert(n == offset(i + 1) - offset(i));
    for (int j = 0; j < n; ++j)
    {
      // FIXME_BOOST would be nice if we could compare tuples
      assert(std::get<0>(l[j]) == std::get<0>(r[j]));
      assert(std::get<1>(l[j]) == std::get<1>(r[j]));
    }
  }
}

#endif
