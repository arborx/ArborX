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

  TraversalPolicy &setBufferSize(int buffer_size)
  {
    _buffer_size = buffer_size;
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
template <typename Tag, typename ExecutionSpace, typename Predicates,
          typename OffsetView, typename OutView>
std::enable_if_t<std::is_same<Tag, SpatialPredicateTag>{}>
allocateAndInititalizeStorage(Tag, ExecutionSpace const &space,
                              Predicates const &predicates, OffsetView &offset,
                              OutView &out, int buffer_size)
{
  using Access = AccessTraits<Predicates, PredicatesTag>;

  auto const n_queries = Access::size(predicates);
  reallocWithoutInitializing(offset, n_queries + 1);

  buffer_size = std::abs(buffer_size);

  Kokkos::deep_copy(space, offset, buffer_size);

  if (buffer_size != 0)
  {
    exclusivePrefixSum(space, offset);

    // Use calculation for the size to avoid calling lastElement(offset) as it
    // will launch an extra kernel to copy to host.
    reallocWithoutInitializing(out, n_queries * buffer_size);
  }
}

template <typename Tag, typename ExecutionSpace, typename Predicates,
          typename OffsetView, typename OutView>
std::enable_if_t<std::is_same<Tag, NearestPredicateTag>{}>
allocateAndInititalizeStorage(Tag, ExecutionSpace const &space,
                              Predicates const &predicates, OffsetView &offset,
                              OutView &out, int /*buffer_size*/)
{
  using Access = AccessTraits<Predicates, PredicatesTag>;

  auto const n_queries = Access::size(predicates);
  reallocWithoutInitializing(offset, n_queries + 1);

  Kokkos::parallel_for(
      "ArborX::BVH::query::nearest::"
      "scan_queries_for_numbers_of_nearest_neighbors",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
      KOKKOS_LAMBDA(int i) { offset(i) = getK(Access::get(predicates, i)); });
  exclusivePrefixSum(space, offset);

  reallocWithoutInitializing(out, lastElement(offset));
}

// Views are passed by reference here because internally Kokkos::realloc()
// is called.
template <typename Tag, typename BVH, typename ExecutionSpace,
          typename Predicates, typename OutputView, typename OffsetView,
          typename Callback>
std::enable_if_t<!is_tagged_post_callback<Callback>{} &&
                 Kokkos::is_view<OutputView>{} && Kokkos::is_view<OffsetView>{}>
queryDispatch(Tag, BVH const &bvh, ExecutionSpace const &space,
              Predicates const &predicates, Callback const &callback,
              OutputView &out, OffsetView &offset,
              Experimental::TraversalPolicy const &policy =
                  Experimental::TraversalPolicy())
{
  using MemorySpace = typename BVH::memory_space;
  using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;

  check_valid_callback(callback, predicates, out);

  auto profiling_prefix =
      std::string("ArborX::BVH::query::") +
      (std::is_same<Tag, SpatialPredicateTag>{} ? "spatial" : "nearest");

  Kokkos::Profiling::pushRegion(profiling_prefix);

  using Access = AccessTraits<Predicates, PredicatesTag>;
  auto const n_queries = Access::size(predicates);

  Kokkos::Profiling::pushRegion(profiling_prefix + "::init_and_alloc");

  allocateAndInititalizeStorage(Tag{}, space, predicates, offset, out,
                                policy._buffer_size);

  Kokkos::Profiling::popRegion();

  auto buffer_status = (std::is_same<Tag, SpatialPredicateTag>{}
                            ? toBufferStatus(policy._buffer_size)
                            : BufferStatus::PreallocationSoft);

  if (policy._sort_predicates)
  {
    Kokkos::Profiling::pushRegion(profiling_prefix + "::compute_permutation");
    auto permute =
        Details::BatchedQueries<DeviceType>::sortQueriesAlongZOrderCurve(
            space, bvh.bounds(), predicates);
    Kokkos::Profiling::popRegion();

    queryImpl(space, BVHParallelTreeTraversal<BVH>{bvh}, predicates, callback,
              out, offset, permute, buffer_status);
  }
  else
  {
    Iota permute;
    queryImpl(space, BVHParallelTreeTraversal<BVH>{bvh}, predicates, callback,
              out, offset, permute, buffer_status);
  }

  Kokkos::Profiling::popRegion();
}

template <typename Tag, typename BVH, typename ExecutionSpace,
          typename Predicates, typename Indices, typename Offset>
inline std::enable_if_t<Kokkos::is_view<Indices>{} && Kokkos::is_view<Offset>{}>
queryDispatch(Tag, BVH const &bvh, ExecutionSpace const &space,
              Predicates const &predicates, Indices &indices, Offset &offset,
              Experimental::TraversalPolicy const &policy =
                  Experimental::TraversalPolicy())
{
  queryDispatch(Tag{}, bvh, space, predicates, DefaultCallback{}, indices,
                offset, policy);
}

template <typename Tag, typename BVH, typename ExecutionSpace,
          typename Predicates, typename OutputView, typename OffsetView,
          typename Callback>
inline std::enable_if_t<is_tagged_post_callback<Callback>{}>
queryDispatch(Tag, BVH const &bvh, ExecutionSpace const &space,
              Predicates const &predicates, Callback const &callback,
              OutputView &out, OffsetView &offset,
              Experimental::TraversalPolicy const &policy =
                  Experimental::TraversalPolicy())
{
  using MemorySpace = typename BVH::memory_space;
  Kokkos::View<int *, MemorySpace> indices("ArborX::BVH::query::indices", 0);
  queryDispatch(Tag{}, bvh, space, predicates, indices, offset, policy);
  callback(predicates, offset, indices, out);
}

template <typename Callback, typename Predicates, typename OutputView>
std::enable_if_t<!Kokkos::is_view<Callback>{} &&
                 !is_tagged_post_callback<Callback>{}>
check_valid_callback_if_first_argument_is_not_a_view(
    Callback const &callback, Predicates const &predicates,
    OutputView const &out)
{
  check_valid_callback(callback, predicates, out);
}

template <typename Callback, typename Predicates, typename OutputView>
std::enable_if_t<!Kokkos::is_view<Callback>{} &&
                 is_tagged_post_callback<Callback>{}>
check_valid_callback_if_first_argument_is_not_a_view(Callback const &,
                                                     Predicates const &,
                                                     OutputView const &)
{
  // TODO
}

template <typename View, typename Predicates, typename OutputView>
std::enable_if_t<Kokkos::is_view<View>{}>
check_valid_callback_if_first_argument_is_not_a_view(View const &,
                                                     Predicates const &,
                                                     OutputView const &)
{
  // do nothing
}

template <typename ExecutionSpace, typename BVH, typename Predicates,
          typename CallbackOrView, typename View, typename... Args>
inline std::enable_if_t<Kokkos::is_view<std::decay_t<View>>{}>
query(ExecutionSpace const &space, BVH const &bvh, Predicates const &predicates,
      CallbackOrView &&callback_or_view, View &&view, Args &&... args)
{
  check_valid_callback_if_first_argument_is_not_a_view(callback_or_view,
                                                       predicates, view);

  using Access = AccessTraits<Predicates, Traits::PredicatesTag>;
  using Tag = typename AccessTraitsHelper<Access>::tag;

  queryDispatch(Tag{}, bvh, space, predicates,
                std::forward<CallbackOrView>(callback_or_view),
                std::forward<View>(view), std::forward<Args>(args)...);
}

template <typename ExecutionSpace, typename BVH, typename Predicates,
          typename Callback>
inline void query(ExecutionSpace const &space, BVH const &bvh,
                  Predicates const &predicates, Callback const &callback,
                  Experimental::TraversalPolicy const &policy =
                      Experimental::TraversalPolicy())
{
  check_valid_callback(callback, predicates);

  using Access = AccessTraits<Predicates, Traits::PredicatesTag>;
  using Tag = typename AccessTraitsHelper<Access>::tag;

  auto profiling_prefix =
      std::string("ArborX::BVH::query::") +
      (std::is_same<Tag, SpatialPredicateTag>{} ? "spatial" : "nearest");

  Kokkos::Profiling::pushRegion(profiling_prefix);

  if (policy._sort_predicates)
  {
    Kokkos::Profiling::pushRegion(profiling_prefix + "::compute_permutation");
    using MemorySpace = typename BVH::memory_space;
    using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;
    auto permute =
        Details::BatchedQueries<DeviceType>::sortQueriesAlongZOrderCurve(
            space, bvh.bounds(), predicates);
    Kokkos::Profiling::popRegion();

    using PermutedPredicates = PermutedData<Predicates, decltype(permute)>;
    traverse(space, bvh, PermutedPredicates{predicates, permute}, callback);
  }
  else
  {
    traverse(space, bvh, predicates, callback);
  }

  Kokkos::Profiling::popRegion();
}

} // namespace BoundingVolumeHierarchyImpl
} // namespace Details
} // namespace ArborX

#endif
