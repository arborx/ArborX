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
  using tag = ArborX::Details::InlineCallbackTag;
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
    ArborX::reallocWithoutInitializing(out, in.extent(0));
    // NOTE woraround to avoid implicit capture of *this
    auto const &points_ = points;
    auto const &origin_ = origin;
    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecutionSpace>(0, n), KOKKOS_LAMBDA(int i) {
          for (int j = offset(i); j < offset(i + 1); ++j)
          {
            out(j) = {in(j), (float)distance(points_(in(j)), origin_)};
          }
        });
  }
};

BOOST_AUTO_TEST_CASE_TEMPLATE(callback_spatial_predicate, TreeTypeTraits,
                              TreeTypeTraitsList)
{
  using Tree = typename TreeTypeTraits::type;
  using ExecutionSpace = typename TreeTypeTraits::execution_space;
  using DeviceType = typename TreeTypeTraits::device_type;

  int const n = 10;
  Kokkos::View<ArborX::Point *, DeviceType> points(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "points"), n);
  ArborX::Point const origin = {{0., 0., 0.}};
  Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace>(0, n),
                       KOKKOS_LAMBDA(int i) {
                         points(i) = {{(double)i, (double)i, (double)i}};
                       });
  auto points_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, points);

  std::vector<Kokkos::pair<int, float>> values;
  values.reserve(n);
  for (int i = 0; i < n; ++i)
    values.emplace_back(i, ArborX::Details::distance(points_host(i), origin));
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
  ArborX::Point const origin = {{0., 0., 0.}};
  Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace>(0, n),
                       KOKKOS_LAMBDA(int i) {
                         points(i) = {{(double)i, (double)i, (double)i}};
                       });
  auto points_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, points);

  std::vector<Kokkos::pair<int, float>> values;
  values.reserve(n);
  for (int i = 0; i < n; ++i)
    values.emplace_back(i, ArborX::Details::distance(points_host(i), origin));
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

#ifndef ARBORX_TEST_DISABLE_CALLBACK_EARLY_EXIT
template <class DeviceType>
struct Experimental_CustomCallbackEarlyExit
{
  Kokkos::View<int *, DeviceType, Kokkos::MemoryTraits<Kokkos::Atomic>> counts;
  template <class Predicate>
  KOKKOS_FUNCTION auto operator()(Predicate const &predicate, int) const
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

struct Experimental_CustomCallbackQuickTraversal
{
  using TreeTraversalControl = ArborX::Experimental::TreeTraversalQuick;
  template <class Predicate, class OutputFunctor>
  KOKKOS_FUNCTION void operator()(Predicate const &predicate, int j,
                                  OutputFunctor const &out) const
  {
    int i = getData(predicate);
    (void)i;
    out(j);
  }
};
static_assert(ArborX::Details::traverse_half<
                  Experimental_CustomCallbackQuickTraversal>::value,
              "");

BOOST_AUTO_TEST_CASE_TEMPLATE(callback_quick_traverse, TreeTypeTraits,
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

  auto b = static_cast<ArborX::Box>(tree.bounds());
  auto predicates = makeIntersectsBoxWithAttachmentQueries<DeviceType, int>(
      {b, b, b, b}, {0, 1, 2, 3});

  Kokkos::View<int *, DeviceType> indices("indices", 0);
  Kokkos::View<int *, DeviceType> offsets("offsets", 0);
  tree.query(ExecutionSpace{}, predicates,
             Experimental_CustomCallbackQuickTraversal{}, indices, offsets);
  BOOST_TEST(
      make_compressed_storage(offsets, indices) ==
          make_reference_solution<int>({1, 2, 3, 2, 3, 3}, {0, 3, 5, 6, 6}),
      tt::per_element());
}
#endif

template <typename DeviceType>
struct CustomInlineCallbackWithAttachment
{
  using tag = ArborX::Details::InlineCallbackTag;
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
    ArborX::reallocWithoutInitializing(out, in.extent(0));
    // NOTE workaround to avoid implicit capture of *this
    auto const &points_ = points;
    auto const &origin_ = origin;
    Kokkos::parallel_for(
        Kokkos::RangePolicy<ExecutionSpace>(0, n), KOKKOS_LAMBDA(int i) {
          auto data_2 = ArborX::getData(queries(i));
          auto data = data_2[1];
          for (int j = offset(i); j < offset(i + 1); ++j)
          {
            out(j) = {in(j), data + (float)distance(points_(in(j)), origin_)};
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

  int const n = 10;
  Kokkos::View<ArborX::Point *, DeviceType> points(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "points"), n);
  ArborX::Point const origin = {{0., 0., 0.}};
  Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace>(0, n),
                       KOKKOS_LAMBDA(int i) {
                         points(i) = {{(double)i, (double)i, (double)i}};
                       });
  auto points_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, points);

  std::vector<Kokkos::pair<int, float>> values;
  values.reserve(n);
  float const delta = 5.f;
  for (int i = 0; i < n; ++i)
    values.emplace_back(
        i, delta + ArborX::Details::distance(points_host(i), origin));
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
  ArborX::Point const origin = {{0., 0., 0.}};
  Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace>(0, n),
                       KOKKOS_LAMBDA(int i) {
                         points(i) = {{(double)i, (double)i, (double)i}};
                       });
  auto points_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, points);

  std::vector<Kokkos::pair<int, float>> values;
  values.reserve(n);
  float const delta = 5.f;
  for (int i = 0; i < n; ++i)
    values.emplace_back(
        i, delta + ArborX::Details::distance(points_host(i), origin));
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
