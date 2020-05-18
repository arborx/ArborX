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
  // the second to actually write down the results at the right location in
  // the flattened array)
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

// This class is the top level query distribution and search algorithm. It is
// implementation specific tree traversal.
// NOTE: There is nothing specific here about spatial, thus one should be able
// to rewrite nearest using the same structure, with a benefit of potentially
// adding threading.
template <typename BVH>
struct BVHParallelTreeTraversal
{
  BVH _bvh;

  template <typename ExecutionSpace, typename Predicates,
            typename InsertGenerator>
  void launch(ExecutionSpace const &space, Predicates const predicates,
              InsertGenerator const &insert_generator) const
  {
    traverse(space, _bvh, predicates, insert_generator);
  }
};

struct Iota
{
  KOKKOS_FUNCTION unsigned int operator()(int const i) const { return i; }
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

  Kokkos::Profiling::pushRegion("ArborX:BVH:spatial_queries");

  using Access = Traits::Access<Predicates, Traits::PredicatesTag>;
  auto const n_queries = Access::size(predicates);

  Kokkos::Profiling::pushRegion("ArborX:BVH:spatial_queries:init_and_alloc");
  reallocWithoutInitializing(offset, n_queries + 1);

  int const buffer_size = std::abs(policy._buffer_size);
  if (buffer_size > 0)
  {
    Kokkos::deep_copy(space, offset, buffer_size);
    exclusivePrefixSum(space, offset);
    // Use calculation for the size to avoid calling lastElement(offset) as it
    // will launch an extra kernel to copy to host. And there is unnecessary to
    // fill with invalid indices.
    reallocWithoutInitializing(out, n_queries * buffer_size);
  }
  else
  {
    Kokkos::deep_copy(offset, 0);
  }
  Kokkos::Profiling::popRegion();

  if (policy._sort_predicates)
  {
    Kokkos::Profiling::pushRegion(
        "ArborX:BVH:spatial_queries:compute_permutation");
    auto permute =
        Details::BatchedQueries<DeviceType>::sortQueriesAlongZOrderCurve(
            space, bvh.bounds(), predicates);
    Kokkos::Profiling::popRegion();

    queryImpl(space, BVHParallelTreeTraversal<BVH>{bvh}, predicates, callback,
              out, offset, permute, toBufferStatus(policy._buffer_size));
  }
  else
  {
    Iota permute;
    queryImpl(space, BVHParallelTreeTraversal<BVH>{bvh}, predicates, callback,
              out, offset, permute, toBufferStatus(policy._buffer_size));
  }

  Kokkos::Profiling::popRegion();
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

  using Access = Traits::Access<Predicates, Traits::PredicatesTag>;
  auto const n_queries = Access::size(predicates);

  bool const use_deprecated_nearest_query_algorithm =
      (policy._traversal_algorithm ==
       NearestQueryAlgorithm::PriorityQueueBased_Deprecated);
  std::ignore = use_deprecated_nearest_query_algorithm;

  Kokkos::Profiling::pushRegion("ArborX:BVH:nearest_queries:init_and_alloc");

  reallocWithoutInitializing(offset, n_queries + 1);
  Kokkos::parallel_for(
      ARBORX_MARK_REGION("scan_queries_for_numbers_of_nearest_neighbors"),
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
      KOKKOS_LAMBDA(int i) { offset(i) = getK(Access::get(predicates, i)); });
  exclusivePrefixSum(space, offset);

  int const n_results = lastElement(offset);
  reallocWithoutInitializing(out, n_results);

  Kokkos::Profiling::popRegion();

  if (policy._sort_predicates)
  {
    Kokkos::Profiling::pushRegion(
        "ArborX:BVH:nearest_queries:compute_permutation");
    auto permute =
        Details::BatchedQueries<DeviceType>::sortQueriesAlongZOrderCurve(
            space, bvh.bounds(), predicates);
    Kokkos::Profiling::popRegion();

    queryImpl(space, BVHParallelTreeTraversal<BVH>{bvh}, predicates, callback,
              out, offset, permute, BufferStatus::PreallocationSoft);
  }
  else
  {
    Iota permute;
    queryImpl(space, BVHParallelTreeTraversal<BVH>{bvh}, predicates, callback,
              out, offset, permute, BufferStatus::PreallocationSoft);
  }

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
