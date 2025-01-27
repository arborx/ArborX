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

#include <boost/test/unit_test.hpp>

#include <numeric>
#include <vector>

#include "Search_UnitTestHelpers.hpp"
// clang-format off
#include "ArborXTest_TreeTypeTraits.hpp"
// clang-format on

BOOST_AUTO_TEST_SUITE(Callbacks)

namespace tt = boost::test_tools;

template <typename Points>
struct CustomInlineCallback
{
  static_assert(Kokkos::is_view_v<Points> && Points::rank() == 1);
  using Point = typename Points::value_type;

  Points points;
  Point const origin = {{0., 0., 0.}};

  template <typename Query, typename Insert>
  KOKKOS_FUNCTION void operator()(Query const &, int index,
                                  Insert const &insert) const
  {
    auto const distance_to_origin =
        ArborX::Details::distance(points(index), origin);
    insert({index, distance_to_origin});
  }
};

template <typename Points>
struct CustomPostCallback
{
  static_assert(Kokkos::is_view_v<Points> && Points::rank() == 1);
  using Point = typename Points::value_type;
  using tag = ArborX::Details::PostCallbackTag;

  Points points;
  Point const origin = {{0., 0., 0.}};
  template <typename Predicates, typename InOutView, typename InView,
            typename OutView>
  void operator()(Predicates const &, InOutView &offset, InView in,
                  OutView &out) const
  {
    using ExecutionSpace = typename Points::execution_space;

    using ArborX::Details::distance;
    auto const n = offset.extent(0) - 1;
    Kokkos::realloc(out, in.extent(0));
    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecutionSpace>(0, n), KOKKOS_CLASS_LAMBDA(int i) {
          for (int j = offset(i); j < offset(i + 1); ++j)
            out(j) = {in(j), distance(points(in(j)), origin)};
        });
  }
};

template <typename Points>
std::vector<Kokkos::pair<int, ArborX::GeometryTraits::coordinate_type_t<
                                  typename Points::value_type>>>
initialize_values(Points const &points, float const delta)
{
  using MemorySpace = typename Points::memory_space;
  using ExecutionSpace = typename Points::execution_space;
  using Coordinate =
      ArborX::GeometryTraits::coordinate_type_t<typename Points::value_type>;
  using PairIndexDistance = Kokkos::pair<int, Coordinate>;

  int const n = points.size();
  Kokkos::View<PairIndexDistance *, MemorySpace> values_device("values_device",
                                                               n);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace>(0, n), KOKKOS_LAMBDA(int i) {
        ArborX::Point const origin{(Coordinate)0, (Coordinate)0, (Coordinate)0};
        values_device(i) = {
            i, delta + ArborX::Details::distance(points(i), origin)};
      });
  std::vector<PairIndexDistance> values(n);
  Kokkos::deep_copy(
      Kokkos::View<PairIndexDistance *, Kokkos::HostSpace>(values.data(), n),
      values_device);
  return values;
}

BOOST_AUTO_TEST_CASE_TEMPLATE(callback_spatial_predicate, TreeTypeTraits,
                              TreeTypeTraitsList)
{
  using Tree = typename TreeTypeTraits::type;
  using ExecutionSpace = typename TreeTypeTraits::execution_space;
  using DeviceType = typename TreeTypeTraits::device_type;

  using Coordinate = ArborX::GeometryTraits::coordinate_type_t<
      typename Tree::bounding_volume_type>;
  using Point = ArborX::Point<3, Coordinate>;
  using Box = ArborX::Box<3, Coordinate>;

  int const n = 10;
  Kokkos::View<Point *, DeviceType> points(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "points"), n);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace>(0, n), KOKKOS_LAMBDA(int i) {
        points(i) = {{(Coordinate)i, (Coordinate)i, (Coordinate)i}};
      });

  auto values = initialize_values(points, /*delta*/ 0.f);
  std::vector<int> offsets = {0, n};

  Tree const tree(ExecutionSpace{}, points);

  ARBORX_TEST_QUERY_TREE_CALLBACK(
      ExecutionSpace{}, tree,
      (makeIntersectsQueries<DeviceType, Box>({
          static_cast<Box>(tree.bounds()),
      })),
      CustomInlineCallback<decltype(points)>{points},
      make_compressed_storage(offsets, values));

  ARBORX_TEST_QUERY_TREE_CALLBACK(ExecutionSpace{}, tree,
                                  (makeIntersectsQueries<DeviceType, Box>({
                                      static_cast<Box>(tree.bounds()),
                                  })),
                                  CustomPostCallback<decltype(points)>{points},
                                  make_compressed_storage(offsets, values));
}

#ifndef ARBORX_TEST_DISABLE_NEAREST_QUERY
BOOST_AUTO_TEST_CASE_TEMPLATE(callback_nearest_predicate, TreeTypeTraits,
                              TreeTypeTraitsList)
{
  using Tree = typename TreeTypeTraits::type;
  using ExecutionSpace = typename TreeTypeTraits::execution_space;
  using DeviceType = typename TreeTypeTraits::device_type;
  using Coordinate = ArborX::GeometryTraits::coordinate_type_t<
      typename Tree::bounding_volume_type>;
  using Point = ArborX::Point<3, Coordinate>;

  int const n = 10;
  Kokkos::View<Point *, DeviceType> points(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "points"), n);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace>(0, n), KOKKOS_LAMBDA(int i) {
        points(i) = {{(Coordinate)i, (Coordinate)i, (Coordinate)i}};
      });
#ifdef KOKKOS_COMPILER_NVCC
  [[maybe_unused]]
#endif
  Point const origin = {{0., 0., 0.}};

  auto values = initialize_values(points, /*delta*/ 0.f);
  std::vector<int> offsets = {0, n};

  Tree const tree(ExecutionSpace{}, points);

  ARBORX_TEST_QUERY_TREE_CALLBACK(
      ExecutionSpace{}, tree,
      (makeNearestQueries<DeviceType, Point>({
          {origin, n},
      })),
      CustomInlineCallback<decltype(points)>{points},
      make_compressed_storage(offsets, values));

  ARBORX_TEST_QUERY_TREE_CALLBACK(ExecutionSpace{}, tree,
                                  (makeNearestQueries<DeviceType, Point>({
                                      {origin, n},
                                  })),
                                  CustomPostCallback<decltype(points)>{points},
                                  make_compressed_storage(offsets, values));
}
#endif

#ifndef ARBORX_TEST_DISABLE_CALLBACK_EARLY_EXIT
template <class DeviceType>
struct Experimental_CustomCallbackEarlyExit
{
  Kokkos::View<int *, DeviceType, Kokkos::MemoryTraits<Kokkos::Atomic>> counts;
  template <class Predicate, typename Value>
  KOKKOS_FUNCTION auto operator()(Predicate const &predicate,
                                  Value const &) const
  {
    int i = getData(predicate);

    if (counts(i)++ < i)
    {
      return ArborX::CallbackTreeTraversalControl::normal_continuation;
    }

    return ArborX::CallbackTreeTraversalControl::early_exit;
  }
};

BOOST_AUTO_TEST_CASE_TEMPLATE(callback_early_exit, TreeTypeTraits,
                              TreeTypeTraitsList)
{
  using Tree = typename TreeTypeTraits::type;
  using ExecutionSpace = typename TreeTypeTraits::execution_space;
  using DeviceType = typename TreeTypeTraits::device_type;
  using BoundingVolume = typename Tree::bounding_volume_type;
  constexpr int DIM = ArborX::GeometryTraits::dimension_v<BoundingVolume>;
  using Coordinate = ArborX::GeometryTraits::coordinate_type_t<BoundingVolume>;
  using Box = ArborX::Box<DIM, Coordinate>;

  auto const tree =
      make<Tree, Box>(ExecutionSpace{}, {
                                            {{{0., 0., 0.}}, {{0., 0., 0.}}},
                                            {{{1., 1., 1.}}, {{1., 1., 1.}}},
                                            {{{2., 2., 2.}}, {{2., 2., 2.}}},
                                            {{{3., 3., 3.}}, {{3., 3., 3.}}},
                                        });

  Kokkos::View<int *, DeviceType> counts("counts", 4);

  std::vector<int> counts_ref(4);
  std::iota(counts_ref.begin(), counts_ref.end(), 1);

  Box b;
  ArborX::Details::expand(b, tree.bounds());
  auto predicates = makeIntersectsWithAttachmentQueries<DeviceType, Box, int>(
      {b, b, b, b}, {0, 1, 2, 3});

  tree.query(ExecutionSpace{}, predicates,
             Experimental_CustomCallbackEarlyExit<DeviceType>{counts});

  auto counts_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, counts);

  BOOST_TEST(counts_host == counts_ref, tt::per_element());
}
#endif

template <typename Points>
struct CustomInlineCallbackWithAttachment
{
  using Point = typename Points::value_type;

  Points points;
  Point const origin = {{0., 0., 0.}};

  template <typename Query, typename Insert>
  KOKKOS_FUNCTION void operator()(Query const &query, int index,
                                  Insert const &insert) const
  {
    auto const distance_to_origin =
        ArborX::Details::distance(points(index), origin);

    auto data = ArborX::getData(query);
    insert({index, data + distance_to_origin});
  }
};
template <typename Points>
struct CustomPostCallbackWithAttachment
{
  using tag = ArborX::Details::PostCallbackTag;

  using Point = typename Points::value_type;

  Points points;
  Point const origin = {{0., 0., 0.}};

  template <typename Predicates, typename InOutView, typename InView,
            typename OutView>
  void operator()(Predicates const &queries, InOutView &offset, InView in,
                  OutView &out) const
  {
    using ExecutionSpace = typename Points::execution_space;
    using ArborX::Details::distance;
    auto const n = offset.extent(0) - 1;
    Kokkos::realloc(out, in.extent(0));
    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecutionSpace>(0, n), KOKKOS_CLASS_LAMBDA(int i) {
          auto data_2 = ArborX::getData(queries(i));
          auto data = data_2[1];
          for (int j = offset(i); j < offset(i + 1); ++j)
          {
            out(j) = {in(j), data + distance(points(in(j)), origin)};
          }
        });
  }
};

BOOST_AUTO_TEST_CASE_TEMPLATE(callback_with_attachment_spatial_predicate,
                              TreeTypeTraits, TreeTypeTraitsList)
{
  using Tree = typename TreeTypeTraits::type;
  using ExecutionSpace = typename TreeTypeTraits::execution_space;
  using DeviceType = typename TreeTypeTraits::device_type;
  using BoundingVolume = typename Tree::bounding_volume_type;
  using Coordinate = ArborX::GeometryTraits::coordinate_type_t<BoundingVolume>;
  using Point = ArborX::Point<3, Coordinate>;
  using Box = ArborX::Box<3, Coordinate>;

  int const n = 10;
  Kokkos::View<Point *, DeviceType> points(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "points"), n);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace>(0, n), KOKKOS_LAMBDA(int i) {
        points(i) = {{(Coordinate)i, (Coordinate)i, (Coordinate)i}};
      });
  Coordinate const delta = 5.f;

  auto values = initialize_values(points, delta);
  std::vector<int> offsets = {0, n};

  Tree const tree(ExecutionSpace{}, points);

  Box bounds;
  ArborX::Details::expand(bounds, tree.bounds());

  ARBORX_TEST_QUERY_TREE_CALLBACK(
      ExecutionSpace{}, tree,
      (makeIntersectsWithAttachmentQueries<DeviceType, Box, Coordinate>(
          {bounds}, {delta})),
      CustomInlineCallbackWithAttachment<decltype(points)>{points},
      make_compressed_storage(offsets, values));

  ARBORX_TEST_QUERY_TREE_CALLBACK(
      ExecutionSpace{}, tree,
      (makeIntersectsWithAttachmentQueries<DeviceType, Box,
                                           Kokkos::Array<Coordinate, 2>>(
          {bounds}, {{0., delta}})),
      CustomPostCallbackWithAttachment<decltype(points)>{points},
      make_compressed_storage(offsets, values));
}

#ifndef ARBORX_TEST_DISABLE_NEAREST_QUERY
BOOST_AUTO_TEST_CASE_TEMPLATE(callback_with_attachment_nearest_predicate,
                              TreeTypeTraits, TreeTypeTraitsList)
{
  using Tree = typename TreeTypeTraits::type;
  using ExecutionSpace = typename TreeTypeTraits::execution_space;
  using DeviceType = typename TreeTypeTraits::device_type;
  using BoundingVolume = typename Tree::bounding_volume_type;
  using Coordinate = ArborX::GeometryTraits::coordinate_type_t<BoundingVolume>;
  using Point = ArborX::Point<3, Coordinate>;

  int const n = 10;
  Kokkos::View<Point *, DeviceType> points(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "points"), n);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace>(0, n), KOKKOS_LAMBDA(int i) {
        points(i) = {{(Coordinate)i, (Coordinate)i, (Coordinate)i}};
      });
#ifdef KOKKOS_COMPILER_NVCC
  [[maybe_unused]]
#endif
  Coordinate const delta = 5.f;
  Point const origin = {{0., 0., 0.}};

  auto values = initialize_values(points, delta);
  std::vector<int> offsets = {0, n};

  Tree const tree(ExecutionSpace{}, points);

  ARBORX_TEST_QUERY_TREE_CALLBACK(
      ExecutionSpace{}, tree,
      (makeNearestWithAttachmentQueries<DeviceType, Point, Coordinate>(
          {{origin, n}}, {delta})),
      CustomInlineCallbackWithAttachment<decltype(points)>{points},
      make_compressed_storage(offsets, values));

  ARBORX_TEST_QUERY_TREE_CALLBACK(
      ExecutionSpace{}, tree,
      (makeNearestWithAttachmentQueries<DeviceType, Point,
                                        Kokkos::Array<Coordinate, 2>>(
          {{origin, n}}, {{0, delta}})),
      CustomPostCallbackWithAttachment<decltype(points)>{points},
      make_compressed_storage(offsets, values));
}
#endif

BOOST_AUTO_TEST_SUITE_END()
