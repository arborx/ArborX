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

#ifndef ARBORX_DETAILS_BOUNDING_VOLUME_HIERARCHY_IMPL_HPP
#define ARBORX_DETAILS_BOUNDING_VOLUME_HIERARCHY_IMPL_HPP

#include <ArborX_DetailsBatchedQueries.hpp>
#include <ArborX_DetailsConcepts.hpp>  // is_detected
#include <ArborX_DetailsKokkosExt.hpp> // ArithmeticTraits
#include <ArborX_DetailsTreeTraversal.hpp>
#include <ArborX_DetailsUtils.hpp>
#include <ArborX_Predicates.hpp>
#include <ArborX_Traits.hpp>

#include <Kokkos_Pair.hpp>
#include <Kokkos_View.hpp>

namespace ArborX
{

template <typename DeviceType>
class BoundingVolumeHierarchy;

namespace Details
{

// archetypal expression for user callbacks
template <typename Callback, typename Predicate, typename Out>
using NearestPredicateInlineCallbackArchetypeExpression =
    decltype(std::declval<Callback const &>()(
        std::declval<Predicate const &>(), 0, 0., std::declval<Out const &>()));

template <typename Callback, typename Predicate, typename Out>
using SpatialPredicateInlineCallbackArchetypeExpression =
    decltype(std::declval<Callback const &>()(std::declval<Predicate const &>(),
                                              0, std::declval<Out const &>()));

// output functor to pass to the callback during detection
template <typename T>
struct Sink
{
  void operator()(T const &) const {}
};

template <typename Predicates>
using PredicatesHelper =
    decay_result_of_get_t<Traits::Access<Predicates, Traits::PredicatesTag>>;

template <typename OutputView>
using OutputFunctorHelper = Sink<typename OutputView::value_type>;

struct InlineCallbackTag
{
};

struct PostCallbackTag
{
};

// Silly name to discourage misuse...
enum class NearestQueryAlgorithm
{
  StackBased_Default,
  PriorityQueueBased_Deprecated
};

struct CallbackDefaultSpatialPredicate
{
  using tag = InlineCallbackTag;
  template <typename Query, typename Insert>
  KOKKOS_FUNCTION void operator()(Query const &, int index,
                                  Insert const &insert) const
  {
    insert(index);
  }
};

struct CallbackDefaultNearestPredicate
{
  using tag = InlineCallbackTag;
  template <typename Query, typename Insert>
  KOKKOS_FUNCTION void operator()(Query const &, int index, float,
                                  Insert const &insert) const
  {
    insert(index);
  }
};

struct CallbackDefaultNearestPredicateWithDistance
{
  using tag = InlineCallbackTag;
  template <typename Query, typename Insert>
  KOKKOS_FUNCTION void operator()(Query const &, int index, float distance,
                                  Insert const &insert) const
  {
    insert({index, distance});
  }
};

template <typename DeviceType>
struct BoundingVolumeHierarchyImpl
{
  // Views are passed by reference here because internally Kokkos::realloc()
  // is called.
  template <typename Predicates>
  static void queryDispatch(SpatialPredicateTag,
                            BoundingVolumeHierarchy<DeviceType> const &bvh,
                            Predicates const &predicates,
                            Kokkos::View<int *, DeviceType> &indices,
                            Kokkos::View<int *, DeviceType> &offset,
                            int buffer_size = 0)
  {
    queryDispatch(SpatialPredicateTag{}, bvh, predicates,
                  CallbackDefaultSpatialPredicate{}, indices, offset,
                  buffer_size);
  }

  template <typename Predicates, typename OutputView, typename Callback>
  static std::enable_if_t<
      std::is_same<typename Callback::tag, InlineCallbackTag>::value>
  queryDispatch(SpatialPredicateTag,
                BoundingVolumeHierarchy<DeviceType> const &bvh,
                Predicates const &predicates, Callback const &callback,
                OutputView &out, Kokkos::View<int *, DeviceType> &offset,
                int buffer_size = 0);

  template <typename Predicates, typename OutputView, typename Callback>
  static std::enable_if_t<
      std::is_same<typename Callback::tag, PostCallbackTag>::value>
  queryDispatch(SpatialPredicateTag,
                BoundingVolumeHierarchy<DeviceType> const &bvh,
                Predicates const &predicates, Callback const &callback,
                OutputView &out, Kokkos::View<int *, DeviceType> &offset,
                int buffer_size = 0)
  {
    Kokkos::View<int *, DeviceType> indices("indices", 0);
    queryDispatch(SpatialPredicateTag{}, bvh, predicates, indices, offset,
                  buffer_size);
    callback(predicates, offset, indices, out);
  }

  template <typename Predicates, typename OutputView, typename Callback>
  static std::enable_if_t<
      std::is_same<typename Callback::tag, InlineCallbackTag>::value>
  queryDispatch(
      NearestPredicateTag, BoundingVolumeHierarchy<DeviceType> const &bvh,
      Predicates const &predicates, Callback const &callback, OutputView &out,
      Kokkos::View<int *, DeviceType> &offset,
      NearestQueryAlgorithm which = NearestQueryAlgorithm::StackBased_Default);

  template <typename Predicates, typename OutputView, typename Callback>
  static std::enable_if_t<
      std::is_same<typename Callback::tag, PostCallbackTag>::value>
  queryDispatch(
      NearestPredicateTag, BoundingVolumeHierarchy<DeviceType> const &bvh,
      Predicates const &predicates, Callback const &callback, OutputView &out,
      Kokkos::View<int *, DeviceType> &offset,
      NearestQueryAlgorithm which = NearestQueryAlgorithm::StackBased_Default)
  {
    Kokkos::View<Kokkos::pair<int, float> *, DeviceType> pairs(
        "pairs_index_distance", 0);
    queryDispatch(NearestPredicateTag{}, bvh, predicates,
                  CallbackDefaultNearestPredicateWithDistance{}, pairs, offset,
                  which);
    callback(predicates, offset, pairs, out);
  }

  template <typename Predicates>
  static void queryDispatch(
      NearestPredicateTag, BoundingVolumeHierarchy<DeviceType> const &bvh,
      Predicates const &predicates, Kokkos::View<int *, DeviceType> &indices,
      Kokkos::View<int *, DeviceType> &offset,
      NearestQueryAlgorithm which = NearestQueryAlgorithm::StackBased_Default)
  {
    queryDispatch(NearestPredicateTag{}, bvh, predicates,
                  CallbackDefaultNearestPredicate{}, indices, offset, which);
  }

  template <typename Predicates>
  static void queryDispatch(
      NearestPredicateTag, BoundingVolumeHierarchy<DeviceType> const &bvh,
      Predicates const &predicates, Kokkos::View<int *, DeviceType> &indices,
      Kokkos::View<int *, DeviceType> &offset,
      Kokkos::View<float *, DeviceType> &distances,
      NearestQueryAlgorithm which = NearestQueryAlgorithm::StackBased_Default)
  {
    Kokkos::View<Kokkos::pair<int, float> *, DeviceType> out(
        "pairs_index_distance", 0);
    queryDispatch(NearestPredicateTag{}, bvh, predicates,
                  CallbackDefaultNearestPredicateWithDistance{}, out, offset,
                  which);
    auto const n = out.extent(0);
    reallocWithoutInitializing(indices, n);
    reallocWithoutInitializing(distances, n);
    using ExecutionSpace = typename DeviceType::execution_space;
    Kokkos::parallel_for(ARBORX_MARK_REGION("split_pairs"),
                         Kokkos::RangePolicy<ExecutionSpace>(0, n),
                         KOKKOS_LAMBDA(int i) {
                           indices(i) = out(i).first;
                           distances(i) = out(i).second;
                         });
  }
};

// The which parameter let the developer chose from two different tree
// traversal algorithms.  With the default argument, the nearest queries are
// performed using a stack.  This was deemed to be slightly more efficient than
// the other alternative that uses a priority queue.  The existence of that
// parameter shall not be advertised to the user.
template <typename DeviceType>
template <typename Predicates, typename OutputView, typename Callback>
std::enable_if_t<std::is_same<typename Callback::tag, InlineCallbackTag>::value>
BoundingVolumeHierarchyImpl<DeviceType>::queryDispatch(
    NearestPredicateTag, BoundingVolumeHierarchy<DeviceType> const &bvh,
    Predicates const &predicates, Callback const &callback, OutputView &out,
    Kokkos::View<int *, DeviceType> &offset, NearestQueryAlgorithm which)
{
  static_assert(is_detected<NearestPredicateInlineCallbackArchetypeExpression,
                            Callback, PredicatesHelper<Predicates>,
                            OutputFunctorHelper<OutputView>>::value,
                "Callback function does not have the correct signature");

  Kokkos::Profiling::pushRegion("ArborX:BVH:nearest_queries");

  using ExecutionSpace = typename DeviceType::execution_space;

  bool const use_deprecated_nearest_query_algorithm =
      which == NearestQueryAlgorithm::PriorityQueueBased_Deprecated;

  using Access = Traits::Access<Predicates, Traits::PredicatesTag>;
  auto const n_queries = Access::size(predicates);

  Kokkos::Profiling::pushRegion("ArborX:BVH:sort_queries");

  auto const permute =
      Details::BatchedQueries<DeviceType>::sortQueriesAlongZOrderCurve(
          bvh.bounds(), predicates);

  // FIXME  readability!  queries is a sorted copy of the predicates
  auto queries = Details::BatchedQueries<DeviceType>::applyPermutation(
      permute, predicates);

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX:BVH:init_offset");

  reallocWithoutInitializing(offset, n_queries + 1);

  Kokkos::parallel_for(
      ARBORX_MARK_REGION("scan_queries_for_numbers_of_nearest_neighbors"),
      Kokkos::RangePolicy<ExecutionSpace>(0, n_queries),
      KOKKOS_LAMBDA(int i) { offset(permute(i)) = getK(queries(i)); });

  exclusivePrefixSum(offset);
  int const n_results = lastElement(offset);

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX:BVH:traversal");

  reallocWithoutInitializing(out, n_results);
  auto tmp_offset = cloneWithoutInitializingNorCopying(offset);
  if (use_deprecated_nearest_query_algorithm)
  {
    Kokkos::parallel_for(
        ARBORX_MARK_REGION("perform_deprecated_nearest_queries"),
        Kokkos::RangePolicy<ExecutionSpace>(0, n_queries),
        KOKKOS_LAMBDA(int i) {
          int count = 0;
          auto const shift = offset(permute(i));
          auto const &query = queries(i);
          Details::TreeTraversal<DeviceType>::query(
              bvh, query,
              [&query, &callback, &out, shift, &count](int index,
                                                       float distance) {
                callback(query, index, distance,
                         [&out, shift, &count](
                             typename OutputView::value_type const &value) {
                           out(shift + count++) = value;
                         });
              });
          tmp_offset(permute(i)) = count;
        });
  }
  else
  {
    // Allocate buffer over which to perform heap operations in
    // TreeTraversal::nearestQuery() to store nearest leaf nodes found
    // so far.  It is not possible to anticipate how much memory to
    // allocate since the number of nearest neighbors k is only known at
    // runtime.
    Kokkos::View<Kokkos::pair<int, float> *, DeviceType> buffer(
        Kokkos::ViewAllocateWithoutInitializing("buffer"), n_results);

    Kokkos::parallel_for(
        ARBORX_MARK_REGION("perform_nearest_queries"),
        Kokkos::RangePolicy<ExecutionSpace>(0, n_queries),
        KOKKOS_LAMBDA(int i) {
          int count = 0;
          auto const shift = offset(permute(i));
          auto const &query = queries(i);
          Details::TreeTraversal<DeviceType>::query(
              bvh, query,
              [&query, &callback, &out, shift, &count](int index,
                                                       float distance) {
                callback(query, index, distance,
                         [&out, shift, &count](
                             typename OutputView::value_type const &value) {
                           out(shift + count++) = value;
                         });
              },
              Kokkos::subview(buffer,
                              Kokkos::make_pair(offset(permute(i)),
                                                offset(permute(i) + 1))));
          tmp_offset(permute(i)) = count;
        });
  }

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX:BVH:filter_out_invalid_entries");

  // Find out if they are any invalid entries in the indices (i.e. at least
  // one query asked for more neighbors than there are leaves in the tree) and
  // eliminate them if necessary.
  exclusivePrefixSum(tmp_offset);
  int const n_tmp_results = lastElement(tmp_offset);
  if (n_tmp_results != n_results)
  {
    OutputView tmp_out(Kokkos::ViewAllocateWithoutInitializing(out.label()),
                       n_tmp_results);

    Kokkos::parallel_for(ARBORX_MARK_REGION("copy_valid_entries"),
                         Kokkos::RangePolicy<ExecutionSpace>(0, n_queries),
                         KOKKOS_LAMBDA(int q) {
                           for (int i = 0;
                                i < tmp_offset(q + 1) - tmp_offset(q); ++i)
                           {
                             tmp_out(tmp_offset(q) + i) = out(offset(q) + i);
                           }
                         });
    out = tmp_out;
    offset = tmp_offset;
  }

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::popRegion();
}

// The buffer_size argument let the user provide an upper bound for the number
// of results per query.  If the guess is accurate, it avoid performing the tree
// traversals twice (the 1st one to count the number of results per query, the
// 2nd to actually write down the results at the right location in the flattened
// array)
// The default value zero disable the buffer optimization.  The sign of the
// integer is used to specify the policy in the case the size insufficient.  If
// it is positive, the code falls back to the default behavior and performs a
// second pass.  If it is negative, it throws an exception.
template <typename DeviceType>
template <typename Predicates, typename OutputView, typename Callback>
std::enable_if_t<std::is_same<typename Callback::tag, InlineCallbackTag>::value>
BoundingVolumeHierarchyImpl<DeviceType>::queryDispatch(
    SpatialPredicateTag, BoundingVolumeHierarchy<DeviceType> const &bvh,
    Predicates const &predicates, Callback const &callback, OutputView &out,
    Kokkos::View<int *, DeviceType> &offset, int buffer_size)
{
  static_assert(is_detected<SpatialPredicateInlineCallbackArchetypeExpression,
                            Callback, PredicatesHelper<Predicates>,
                            OutputFunctorHelper<OutputView>>::value,
                "Callback function does not have the correct signature");

  Kokkos::Profiling::pushRegion("ArborX:BVH:spatial_queries");

  using ExecutionSpace = typename DeviceType::execution_space;

  using Access = Traits::Access<Predicates, Traits::PredicatesTag>;
  auto const n_queries = Access::size(predicates);

  bool const throw_if_buffer_optimization_fails = (buffer_size < 0);
  buffer_size = std::abs(buffer_size);

  Kokkos::Profiling::pushRegion("ArborX:BVH:sort_queries");

  auto const permute =
      Details::BatchedQueries<DeviceType>::sortQueriesAlongZOrderCurve(
          bvh.bounds(), predicates);

  // FIXME  readability!  queries is a sorted copy of the predicates
  auto queries = Details::BatchedQueries<DeviceType>::applyPermutation(
      permute, predicates);

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX:BVH:first_pass");

  // Initialize view
  // [ 0 0 0 .... 0 0 ]
  //                ^
  //                N
  reallocWithoutInitializing(offset, n_queries + 1);

  // Say we found exactly two object for each query:
  // [ 2 2 2 .... 2 0 ]
  //   ^            ^
  //   0th          Nth element in the view
  if (buffer_size > 0)
  {
    reallocWithoutInitializing(out, n_queries * buffer_size);
    // NOTE I considered filling with invalid indices but it is unecessary
    // work

    Kokkos::parallel_for(
        ARBORX_MARK_REGION("first_pass_at_the_search_with_buffer_optimization"),
        Kokkos::RangePolicy<ExecutionSpace>(0, n_queries),
        KOKKOS_LAMBDA(int i) {
          int count = 0;
          auto const shift = permute(i) * buffer_size;
          auto const &query = queries(i);
          Details::TreeTraversal<DeviceType>::query(
              bvh, query,
              [&query, &callback, buffer_size, &out, shift, &count](int index) {
                if (count < buffer_size)
                  callback(query, index,
                           [&count, &out, shift](
                               typename OutputView::value_type const &value) {
                             out(shift + count++) = value;
                           });
                else
                  callback(query, index,
                           [&count](typename OutputView::value_type const &) {
                             ++count;
                           });
              });
          offset(permute(i)) = count;
        });
  }
  else
  {
    Kokkos::parallel_for(
        ARBORX_MARK_REGION(
            "first_pass_at_the_search_count_the_number_of_values"),
        Kokkos::RangePolicy<ExecutionSpace>(0, n_queries),
        KOKKOS_LAMBDA(int i) {
          int count = 0;
          auto const &query = queries(i);
          Details::TreeTraversal<DeviceType>::query(
              bvh, query, [&query, &callback, &count](int index) {
                callback(query, index,
                         [&count](typename OutputView::value_type const &) {
                           ++count;
                         });
              });
          offset(permute(i)) = count;
        });
  }

  // NOTE max() internally calls Kokkos::parallel_reduce.  Only pay for it if
  // actually trying buffer optimization. In principle, any strictly
  // positive value can be assigned otherwise.
  auto const max_results_per_query =
      (buffer_size > 0)
          ? max(Kokkos::subview(offset,
                                Kokkos::pair<size_t, size_t>(0, n_queries)))
          : std::numeric_limits<typename std::remove_reference<decltype(
                offset)>::type::value_type>::max();

  // Then we would get:
  // [ 0 2 4 .... 2N-2 2N ]
  //                    ^
  //                    N
  exclusivePrefixSum(offset);

  // Let us extract the last element in the view which is the total count of
  // objects which where found to meet the query predicates:
  //
  // [ 2N ]
  int const n_results = lastElement(offset);

  Kokkos::Profiling::popRegion();

  // Exit early if either no results were found for any of the queries, or
  // nothing was inserted inside a callback for found results. This check
  // guarantees that the second pass will not be executed independent of
  // buffer_size.
  if (n_results == 0)
  {
    Kokkos::Profiling::popRegion();
    return;
  }

  if (max_results_per_query > buffer_size)
  {
    Kokkos::Profiling::pushRegion("ArborX:BVH:second_pass");

    // FIXME can definitely do better about error message
    ARBORX_ASSERT(!throw_if_buffer_optimization_fails);

    // We allocate the memory and fill
    //
    // [ A0 A1 B0 B1 C0 C1 ... X0 X1 ]
    //   ^     ^     ^         ^     ^
    //   0     2     4         2N-2  2N
    reallocWithoutInitializing(out, n_results);
    Kokkos::parallel_for(
        ARBORX_MARK_REGION("second_pass"),
        Kokkos::RangePolicy<ExecutionSpace>(0, n_queries),
        KOKKOS_LAMBDA(int i) {
          int count = 0;
          auto const shift = offset(permute(i));
          auto const &query = queries(i);
          Details::TreeTraversal<DeviceType>::query(
              bvh, query, [&query, &callback, &out, shift, &count](int index) {
                callback(query, index,
                         [&count, &out,
                          shift](typename OutputView::value_type const &value) {
                           out(shift + count++) = value;
                         });
              });
          assert(offset(permute(i) + 1) - offset(permute(i)) == count);
        });

    Kokkos::Profiling::popRegion();
  }
  // do not copy if by some miracle each query exactly yielded as many results
  // as the buffer size
  else if (n_results != static_cast<int>(n_queries) * buffer_size)
  {
    Kokkos::Profiling::pushRegion("ArborX:BVH:copy_values");

    OutputView tmp_out(Kokkos::ViewAllocateWithoutInitializing(out.label()),
                       n_results);
    Kokkos::parallel_for(ARBORX_MARK_REGION("copy_valid_values"),
                         Kokkos::RangePolicy<ExecutionSpace>(0, n_queries),
                         KOKKOS_LAMBDA(int q) {
                           for (int i = 0; i < offset(q + 1) - offset(q); ++i)
                           {
                             tmp_out(offset(q) + i) = out(q * buffer_size + i);
                           }
                         });
    out = tmp_out;

    Kokkos::Profiling::popRegion();
  }
  Kokkos::Profiling::popRegion();
}

} // namespace Details
} // namespace ArborX

#endif
