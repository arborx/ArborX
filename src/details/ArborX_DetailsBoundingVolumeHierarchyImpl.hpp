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

#include <ArborX_AccessTraits.hpp>
#include <ArborX_Callbacks.hpp>
#include <ArborX_DetailsBatchedQueries.hpp>
#include <ArborX_DetailsBufferOptimization.hpp>
#include <ArborX_DetailsConcepts.hpp>  // is_detected
#include <ArborX_DetailsKokkosExt.hpp> // ArithmeticTraits
#include <ArborX_DetailsTreeTraversal.hpp>
#include <ArborX_DetailsUtils.hpp>
#include <ArborX_Predicates.hpp>

#include <Kokkos_Core.hpp>

namespace ArborX
{
namespace Details
{

// Silly name to discourage misuse...
enum class NearestQueryAlgorithm
{
  StackBased_Default,
  PriorityQueueBased_Deprecated
};

} // namespace Details

namespace Experimental
{
struct TraversalPolicy
{
  // Buffer size lets a user provide an upper bound for the number of results
  // per query. If the guess is accurate, it avoids performing the tree
  // traversals twice (the first one to count the number of results per query,
  // the second to actually write down the results at the right location in the
  // flattened array)
  //
  // The default value zero disables the buffer optimization. The sign of the
  // integer is used to specify the policy in the case the size insufficient.
  // If it is positive, the code falls back to the default behavior and
  // performs a second pass. If it is negative, it throws an exception.
  int _buffer_size = 0;

  // Sort predicates allows disabling predicate sorting.
  bool _sort_predicates = true;

  // This parameter lets the developer choose from two different tree
  // traversal algorithms. With the default argument, the nearest queries are
  // performed using a stack. This was deemed to be slightly more efficient
  // than the other alternative that uses a priority queue. The existence of
  // the parameter shall not be advertised to the user.
  Details::NearestQueryAlgorithm _traversal_algorithm =
      Details::NearestQueryAlgorithm::StackBased_Default;

  TraversalPolicy &setBufferSize(int buffer_size)
  {
    _buffer_size = buffer_size;
    return *this;
  }
  TraversalPolicy &
  setTraversalAlgorithm(Details::NearestQueryAlgorithm traversal_algorithm)
  {
    _traversal_algorithm = traversal_algorithm;
    return *this;
  }
  TraversalPolicy &setPredicateSorting(bool sort_predicates)
  {
    _sort_predicates = sort_predicates;
    return *this;
  }
};

} // namespace Experimental

namespace Details
{

template <typename BVH>
struct WrappedBVH
{
  BVH bvh_;

  template <typename ExecutionSpace, typename Predicates, typename Callback>
  void operator()(ExecutionSpace const &space, Predicates const predicates,
                  Callback const &callback) const
  {
    traverse(space, bvh_, predicates, callback);
  }
};

namespace BoundingVolumeHierarchyImpl
{
// Views are passed by reference here because internally Kokkos::realloc()
// is called.
template <typename BVH, typename ExecutionSpace, typename Predicates,
          typename OutputView, typename OffsetView, typename Callback>
std::enable_if_t<std::is_same<typename Callback::tag, InlineCallbackTag>::value>
queryDispatch(SpatialPredicateTag, BVH const &bvh, ExecutionSpace const &space,
              Predicates const &predicates, Callback const &callback,
              OutputView &out, OffsetView &offset,
              Experimental::TraversalPolicy const &policy =
                  Experimental::TraversalPolicy())
{
  using MemorySpace = typename BVH::memory_space;
  using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;

  check_valid_callback(callback, predicates, out);

  if (policy._sort_predicates)
  {
    auto permute =
        Details::BatchedQueries<DeviceType>::sortQueriesAlongZOrderCurve(
            space, bvh.bounds(), predicates);
    auto permuted_predicates =
        Details::BatchedQueries<DeviceType>::applyPermutation(space, permute,
                                                              predicates);
    Kokkos::resize(permute, permute.size() + 1);
    Kokkos::deep_copy(Kokkos::subview(permute, permute.size() - 1),
                      permute.size() - 1);
    using Access = Traits::Access<Predicates, Traits::PredicatesTag>;
    auto const n_queries = Access::size(predicates);
    reallocWithoutInitializing(offset, n_queries + 1);
    auto permuted_offset = makePermutedView(permute, offset);
    queryImpl(space, WrappedBVH<BVH>{bvh}, permuted_predicates, callback, out,
              permuted_offset, policy._buffer_size);
  }
  else
  {
    using Access = Traits::Access<Predicates, Traits::PredicatesTag>;
    auto const n_queries = Access::size(predicates);
    reallocWithoutInitializing(offset, n_queries + 1);

    queryImpl(space, WrappedBVH<BVH>{bvh}, predicates, callback, out, offset,
              policy._buffer_size);
  }
}

template <typename BVH, typename ExecutionSpace, typename Predicates,
          typename Indices, typename Offset>
inline std::enable_if_t<Kokkos::is_view<Indices>{} && Kokkos::is_view<Offset>{}>
queryDispatch(SpatialPredicateTag, BVH const &bvh, ExecutionSpace const &space,
              Predicates const &predicates, Indices &indices, Offset &offset,
              Experimental::TraversalPolicy const &policy =
                  Experimental::TraversalPolicy())
{
  queryDispatch(SpatialPredicateTag{}, bvh, space, predicates,
                CallbackDefaultSpatialPredicate{}, indices, offset, policy);
}

template <typename BVH, typename ExecutionSpace, typename Predicates,
          typename OutputView, typename OffsetView, typename Callback>
inline std::enable_if_t<
    std::is_same<typename Callback::tag, PostCallbackTag>::value>
queryDispatch(SpatialPredicateTag, BVH const &bvh, ExecutionSpace const &space,
              Predicates const &predicates, Callback const &callback,
              OutputView &out, OffsetView &offset,
              Experimental::TraversalPolicy const &policy =
                  Experimental::TraversalPolicy())
{
  using MemorySpace = typename BVH::memory_space;
  Kokkos::View<int *, MemorySpace> indices("indices", 0);
  queryDispatch(SpatialPredicateTag{}, bvh, space, predicates, indices, offset,
                policy);
  callback(predicates, offset, indices, out);
}

template <typename BVH, typename ExecutionSpace, typename Predicates,
          typename OutputView, typename OffsetView, typename Callback>
std::enable_if_t<std::is_same<typename Callback::tag, InlineCallbackTag>::value>
queryDispatch(NearestPredicateTag, BVH const &bvh, ExecutionSpace const &space,
              Predicates const &predicates, Callback const &callback,
              OutputView &out, OffsetView &offset,
              Experimental::TraversalPolicy const &policy =
                  Experimental::TraversalPolicy())
{
  using MemorySpace = typename BVH::memory_space;
  using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;

  check_valid_callback(callback, predicates, out);

  Kokkos::Profiling::pushRegion("ArborX:BVH:nearest_queries");

  bool const use_deprecated_nearest_query_algorithm =
      (policy._traversal_algorithm ==
       NearestQueryAlgorithm::PriorityQueueBased_Deprecated);

  using Access = Traits::Access<Predicates, Traits::PredicatesTag>;
  auto const n_queries = Access::size(predicates);

  Kokkos::Profiling::pushRegion("ArborX:BVH:sort_queries");

  Kokkos::View<unsigned int *, MemorySpace> permute;
  if (policy._sort_predicates)
  {
    permute = Details::BatchedQueries<DeviceType>::sortQueriesAlongZOrderCurve(
        space, bvh.bounds(), predicates);
  }
  else
  {
    permute = Kokkos::View<unsigned int *, MemorySpace>(
        Kokkos::ViewAllocateWithoutInitializing("permute"), n_queries);
    iota(space, permute);
  }

  // FIXME  readability!  queries is a sorted copy of the predicates
  auto queries = Details::BatchedQueries<DeviceType>::applyPermutation(
      space, permute, predicates);

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX:BVH:init_offset");

  reallocWithoutInitializing(offset, n_queries + 1);

  Kokkos::parallel_for(
      ARBORX_MARK_REGION("scan_queries_for_numbers_of_nearest_neighbors"),
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
      KOKKOS_LAMBDA(int i) { offset(permute(i)) = getK(queries(i)); });

  exclusivePrefixSum(space, offset);
  int const n_results = lastElement(offset);

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX:BVH:traversal");

  reallocWithoutInitializing(out, n_results);
  auto tmp_offset = cloneWithoutInitializingNorCopying(offset);
  if (use_deprecated_nearest_query_algorithm)
  {
    Kokkos::parallel_for(
        ARBORX_MARK_REGION("perform_deprecated_nearest_queries"),
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
        KOKKOS_LAMBDA(int i) {
          int count = 0;
          auto const shift = offset(permute(i));
          auto const &query = queries(i);
          Details::DeprecatedTreeTraversal<BVH>::query(
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
    Kokkos::View<Kokkos::pair<int, float> *, MemorySpace> buffer(
        Kokkos::ViewAllocateWithoutInitializing("buffer"), n_results);

    Kokkos::parallel_for(
        ARBORX_MARK_REGION("perform_nearest_queries"),
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
        KOKKOS_LAMBDA(int i) {
          int count = 0;
          auto const shift = offset(permute(i));
          auto const &query = queries(i);
          Details::DeprecatedTreeTraversal<BVH>::query(
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
  exclusivePrefixSum(space, tmp_offset);
  int const n_tmp_results = lastElement(tmp_offset);
  if (n_tmp_results != n_results)
  {
    OutputView tmp_out(Kokkos::ViewAllocateWithoutInitializing(out.label()),
                       n_tmp_results);

    Kokkos::parallel_for(
        ARBORX_MARK_REGION("copy_valid_entries"),
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
        KOKKOS_LAMBDA(int q) {
          for (int i = 0; i < tmp_offset(q + 1) - tmp_offset(q); ++i)
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

template <typename BVH, typename ExecutionSpace, typename Predicates,
          typename OutputView, typename OffsetView, typename Callback>
inline std::enable_if_t<
    std::is_same<typename Callback::tag, PostCallbackTag>::value>
queryDispatch(NearestPredicateTag, BVH const &bvh, ExecutionSpace const &space,
              Predicates const &predicates, Callback const &callback,
              OutputView &out, OffsetView &offset,
              Experimental::TraversalPolicy const &policy =
                  Experimental::TraversalPolicy())
{
  using MemorySpace = typename BVH::memory_space;
  Kokkos::View<Kokkos::pair<int, float> *, MemorySpace> pairs(
      "pairs_index_distance", 0);
  queryDispatch(NearestPredicateTag{}, bvh, space, predicates,
                CallbackDefaultNearestPredicateWithDistance{}, pairs, offset,
                policy);
  callback(predicates, offset, pairs, out);
}

template <typename BVH, typename ExecutionSpace, typename Predicates,
          typename Indices, typename Offset>
inline std::enable_if_t<Kokkos::is_view<Indices>{} && Kokkos::is_view<Offset>{}>
queryDispatch(NearestPredicateTag, BVH const &bvh, ExecutionSpace const &space,
              Predicates const &predicates, Indices &indices, Offset &offset,
              Experimental::TraversalPolicy const &policy =
                  Experimental::TraversalPolicy())
{
  queryDispatch(NearestPredicateTag{}, bvh, space, predicates,
                CallbackDefaultNearestPredicate{}, indices, offset, policy);
}

template <typename BVH, typename ExecutionSpace, typename Predicates,
          typename Indices, typename Offset, typename Distances>
inline std::enable_if_t<Kokkos::is_view<Indices>{} &&
                        Kokkos::is_view<Offset>{} &&
                        Kokkos::is_view<Distances>{}>
queryDispatch(NearestPredicateTag, BVH const &bvh, ExecutionSpace const &space,
              Predicates const &predicates, Indices &indices, Offset &offset,
              Distances &distances,
              Experimental::TraversalPolicy const &policy =
                  Experimental::TraversalPolicy())
{
  using MemorySpace = typename BVH::memory_space;
  Kokkos::View<Kokkos::pair<int, float> *, MemorySpace> out(
      "pairs_index_distance", 0);
  queryDispatch(NearestPredicateTag{}, bvh, space, predicates,
                CallbackDefaultNearestPredicateWithDistance{}, out, offset,
                policy);
  auto const n = out.extent(0);
  reallocWithoutInitializing(indices, n);
  reallocWithoutInitializing(distances, n);
  Kokkos::parallel_for(ARBORX_MARK_REGION("split_pairs"),
                       Kokkos::RangePolicy<ExecutionSpace>(space, 0, n),
                       KOKKOS_LAMBDA(int i) {
                         indices(i) = out(i).first;
                         distances(i) = out(i).second;
                       });
}
} // namespace BoundingVolumeHierarchyImpl

template <typename Callback, typename Predicates, typename OutputView>
std::enable_if_t<!Kokkos::is_view<Callback>{}>
check_valid_callback_if_first_argument_is_not_a_view(
    Callback const &callback, Predicates const &predicates,
    OutputView const &out)
{
  check_valid_callback(callback, predicates, out);
}

template <typename View, typename Predicates, typename OutputView>
std::enable_if_t<Kokkos::is_view<View>{}>
check_valid_callback_if_first_argument_is_not_a_view(View const &,
                                                     Predicates const &,
                                                     OutputView const &)
{
  // do nothing
}

} // namespace Details
} // namespace ArborX

#endif
