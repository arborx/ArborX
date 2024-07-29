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

template <typename DeviceType>
struct CustomInlineCallback
{
  Kokkos::View<ArborX::Point *, DeviceType> points;
  ArborX::Point const origin = {{0., 0., 0.}};
  template <typename Query, typename Insert>
  KOKKOS_FUNCTION void operator()(Query const &, int index,
                                  Insert const &insert) const
  {
    float const distance_to_origin =
        ArborX::Details::distance(points(index), origin);
    insert({index, distance_to_origin});
  }
};

template <typename DeviceType>
struct CustomPostCallback
{
  using tag = ArborX::Details::PostCallbackTag;
  Kokkos::View<ArborX::Point *, DeviceType> points;
  ArborX::Point const origin = {{0., 0., 0.}};
  template <typename Predicates, typename InOutView, typename InView,
            typename OutView>
  void operator()(Predicates const &, InOutView &offset, InView in,
                  OutView &out) const
  {
    using ExecutionSpace = typename DeviceType::execution_space;
    using ArborX::Details::distance;
    auto const n = offset.extent(0) - 1;
    Kokkos::realloc(out, in.extent(0));
    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecutionSpace>(0, n), KOKKOS_CLASS_LAMBDA(int i) {
          for (int j = offset(i); j < offset(i + 1); ++j)
          {
            out(j) = {in(j), (float)distance(points(in(j)), origin)};
          }
        });
  }
};

template <typename View>
std::vector<Kokkos::pair<int, float>> initialize_values(View const &points,
                                                        float const delta)
{
  using MemorySpace = typename View::memory_space;
  using ExecutionSpace = typename View::execution_space;
  int const n = points.size();
  Kokkos::View<Kokkos::pair<int, float> *, MemorySpace> values_device(
      "values_device", n);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace>(0, n), KOKKOS_LAMBDA(int i) {
        ArborX::Point const origin = {{0., 0., 0.}};
        values_device(i) = {
            i, delta + ArborX::Details::distance(points(i), origin)};
      });
  std::vector<Kokkos::pair<int, float>> values(n);
  Kokkos::deep_copy(Kokkos::View<Kokkos::pair<int, float> *, Kokkos::HostSpace>(
                        values.data(), n),
                    values_device);
  return values;
}

#ifndef ARBORX_TEST_DISABLE_SPATIAL_QUERY_INTERSECTS_BOX
BOOST_AUTO_TEST_CASE_TEMPLATE(callback_spatial_predicate, TreeTypeTraits,
                              TreeTypeTraitsList)
{
  using Tree = typename TreeTypeTraits::type;
  using ExecutionSpace = typename TreeTypeTraits::execution_space;
  using DeviceType = typename TreeTypeTraits::device_type;

  int const n = 10;
  Kokkos::View<ArborX::Point *, DeviceType> points(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "points"), n);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace>(0, n), KOKKOS_LAMBDA(int i) {
        points(i) = {{(float)i, (float)i, (float)i}};
      });

  auto values = initialize_values(points, /*delta*/ 0.f);
  std::vector<int> offsets = {0, n};

  Tree const tree(ExecutionSpace{}, points);

  ARBORX_TEST_QUERY_TREE_CALLBACK(ExecutionSpace{}, tree,
                                  makeIntersectsBoxQueries<DeviceType>({
                                      static_cast<ArborX::Box>(tree.bounds()),
                                  }),
                                  CustomInlineCallback<DeviceType>{points},
                                  make_compressed_storage(offsets, values));

  ARBORX_TEST_QUERY_TREE_CALLBACK(ExecutionSpace{}, tree,
                                  makeIntersectsBoxQueries<DeviceType>({
                                      static_cast<ArborX::Box>(tree.bounds()),
                                  }),
                                  CustomPostCallback<DeviceType>{points},
                                  make_compressed_storage(offsets, values));
}
#endif

#ifndef ARBORX_TEST_DISABLE_NEAREST_QUERY
BOOST_AUTO_TEST_CASE_TEMPLATE(callback_nearest_predicate, TreeTypeTraits,
                              TreeTypeTraitsList)
{
  using Tree = typename TreeTypeTraits::type;
  using ExecutionSpace = typename TreeTypeTraits::execution_space;
  using DeviceType = typename TreeTypeTraits::device_type;

  int const n = 10;
  Kokkos::View<ArborX::Point *, DeviceType> points(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "points"), n);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace>(0, n), KOKKOS_LAMBDA(int i) {
        points(i) = {{(float)i, (float)i, (float)i}};
      });
  ArborX::Point const origin = {{0., 0., 0.}};

  auto values = initialize_values(points, /*delta*/ 0.f);
  std::vector<int> offsets = {0, n};

  Tree const tree(ExecutionSpace{}, points);

  ARBORX_TEST_QUERY_TREE_CALLBACK(ExecutionSpace{}, tree,
                                  makeNearestQueries<DeviceType>({
                                      {origin, n},
                                  }),
                                  CustomInlineCallback<DeviceType>{points},
                                  make_compressed_storage(offsets, values));

  ARBORX_TEST_QUERY_TREE_CALLBACK(ExecutionSpace{}, tree,
                                  makeNearestQueries<DeviceType>({
                                      {origin, n},
                                  }),
                                  CustomPostCallback<DeviceType>{points},
                                  make_compressed_storage(offsets, values));
}
#endif

#if !defined(ARBORX_TEST_DISABLE_CALLBACK_EARLY_EXIT) &&                       \
    !defined(ARBORX_TEST_DISABLE_SPATIAL_QUERY_INTERSECTS_BOX)
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

  auto const tree =
      make<Tree>(ExecutionSpace{}, {
                                       {{{0., 0., 0.}}, {{0., 0., 0.}}},
                                       {{{1., 1., 1.}}, {{1., 1., 1.}}},
                                       {{{2., 2., 2.}}, {{2., 2., 2.}}},
                                       {{{3., 3., 3.}}, {{3., 3., 3.}}},
                                   });

  Kokkos::View<int *, DeviceType> counts("counts", 4);

  std::vector<int> counts_ref(4);
  std::iota(counts_ref.begin(), counts_ref.end(), 1);

  auto b = static_cast<ArborX::Box>(tree.bounds());
  auto predicates = makeIntersectsBoxWithAttachmentQueries<DeviceType, int>(
      {b, b, b, b}, {0, 1, 2, 3});

  tree.query(ExecutionSpace{}, predicates,
             Experimental_CustomCallbackEarlyExit<DeviceType>{counts});

  auto counts_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, counts);

  BOOST_TEST(counts_host == counts_ref, tt::per_element());
}
#endif

template <typename DeviceType>
struct CustomInlineCallbackWithAttachment
{
  Kokkos::View<ArborX::Point *, DeviceType> points;
  ArborX::Point const origin = {{0., 0., 0.}};
  template <typename Query, typename Insert>
  KOKKOS_FUNCTION void operator()(Query const &query, int index,
                                  Insert const &insert) const
  {
    float const distance_to_origin =
        ArborX::Details::distance(points(index), origin);

    auto data = ArborX::getData(query);
    insert({index, data + distance_to_origin});
  }
};
template <typename DeviceType>
struct CustomPostCallbackWithAttachment
{
  using tag = ArborX::Details::PostCallbackTag;
  Kokkos::View<ArborX::Point *, DeviceType> points;
  ArborX::Point const origin = {{0., 0., 0.}};
  template <typename Predicates, typename InOutView, typename InView,
            typename OutView>
  void operator()(Predicates const &queries, InOutView &offset, InView in,
                  OutView &out) const
  {
    using ExecutionSpace = typename DeviceType::execution_space;
    using ArborX::Details::distance;
    auto const n = offset.extent(0) - 1;
    Kokkos::realloc(out, in.extent(0));
    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecutionSpace>(0, n), KOKKOS_CLASS_LAMBDA(int i) {
          auto data_2 = ArborX::getData(queries(i));
          auto data = data_2[1];
          for (int j = offset(i); j < offset(i + 1); ++j)
          {
            out(j) = {in(j), data + (float)distance(points(in(j)), origin)};
          }
        });
  }
};

#ifndef ARBORX_TEST_DISABLE_SPATIAL_QUERY_INTERSECTS_BOX
BOOST_AUTO_TEST_CASE_TEMPLATE(callback_with_attachment_spatial_predicate,
                              TreeTypeTraits, TreeTypeTraitsList)
{
  using Tree = typename TreeTypeTraits::type;
  using ExecutionSpace = typename TreeTypeTraits::execution_space;
  using DeviceType = typename TreeTypeTraits::device_type;

  int const n = 10;
  Kokkos::View<ArborX::Point *, DeviceType> points(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "points"), n);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace>(0, n), KOKKOS_LAMBDA(int i) {
        points(i) = {{(float)i, (float)i, (float)i}};
      });
  float const delta = 5.f;

  auto values = initialize_values(points, delta);
  std::vector<int> offsets = {0, n};

  Tree const tree(ExecutionSpace{}, points);

  ARBORX_TEST_QUERY_TREE_CALLBACK(
      ExecutionSpace{}, tree,
      (makeIntersectsBoxWithAttachmentQueries<DeviceType, float>(
          {static_cast<ArborX::Box>(tree.bounds())}, {delta})),
      CustomInlineCallbackWithAttachment<DeviceType>{points},
      make_compressed_storage(offsets, values));

  ARBORX_TEST_QUERY_TREE_CALLBACK(
      ExecutionSpace{}, tree,
      (makeIntersectsBoxWithAttachmentQueries<DeviceType,
                                              Kokkos::Array<float, 2>>(
          {static_cast<ArborX::Box>(tree.bounds())}, {{0., delta}})),
      CustomPostCallbackWithAttachment<DeviceType>{points},
      make_compressed_storage(offsets, values));
}
#endif

#ifndef ARBORX_TEST_DISABLE_NEAREST_QUERY
BOOST_AUTO_TEST_CASE_TEMPLATE(callback_with_attachment_nearest_predicate,
                              TreeTypeTraits, TreeTypeTraitsList)
{
  using Tree = typename TreeTypeTraits::type;
  using ExecutionSpace = typename TreeTypeTraits::execution_space;
  using DeviceType = typename TreeTypeTraits::device_type;

  int const n = 10;
  Kokkos::View<ArborX::Point *, DeviceType> points(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "points"), n);
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace>(0, n), KOKKOS_LAMBDA(int i) {
        points(i) = {{(float)i, (float)i, (float)i}};
      });
  float const delta = 5.f;
  ArborX::Point const origin = {{0., 0., 0.}};

  auto values = initialize_values(points, delta);
  std::vector<int> offsets = {0, n};

  Tree const tree(ExecutionSpace{}, points);

  ARBORX_TEST_QUERY_TREE_CALLBACK(
      ExecutionSpace{}, tree,
      (makeNearestWithAttachmentQueries<DeviceType, float>({{origin, n}},
                                                           {delta})),
      CustomInlineCallbackWithAttachment<DeviceType>{points},
      make_compressed_storage(offsets, values));

  ARBORX_TEST_QUERY_TREE_CALLBACK(
      ExecutionSpace{}, tree,
      (makeNearestWithAttachmentQueries<DeviceType, Kokkos::Array<float, 2>>(
          {{origin, n}}, {{0, delta}})),
      CustomPostCallbackWithAttachment<DeviceType>{points},
      make_compressed_storage(offsets, values));
}
#endif

BOOST_AUTO_TEST_SUITE_END()
