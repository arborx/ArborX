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

#ifndef ARBORX_DETAIL_CRS_GRAPH_WRAPPER_IMPL_HPP
#define ARBORX_DETAIL_CRS_GRAPH_WRAPPER_IMPL_HPP

#include <ArborX_AccessTraits.hpp>
#include <ArborX_Callbacks.hpp>
#include <ArborX_DetailsBatchedQueries.hpp>
#include <ArborX_DetailsPermutedData.hpp>
#include <ArborX_Predicates.hpp>
#include <ArborX_TraversalPolicy.hpp>

namespace ArborX
{
namespace Details
{

enum BufferStatus
{
  PreallocationNone = 0,
  PreallocationHard = -1,
  PreallocationSoft = 1
};

inline BufferStatus toBufferStatus(int buffer_size)
{
  if (buffer_size == 0)
    return BufferStatus::PreallocationNone;
  if (buffer_size > 0)
    return BufferStatus::PreallocationSoft;
  return BufferStatus::PreallocationHard;
}

struct FirstPassTag
{
};
struct FirstPassNoBufferOptimizationTag
{
};
struct SecondPassTag
{
};

template <typename PassTag, typename Predicates, typename Callback,
          typename OutputView, typename CountView, typename PermutedOffset>
struct InsertGenerator
{
  Callback _callback;
  OutputView _out;
  CountView _counts;
  PermutedOffset _permuted_offset;

  using ValueType = typename OutputView::value_type;
  using Access = AccessTraits<Predicates, PredicatesTag>;
  using Tag = typename AccessTraitsHelper<Access>::tag;
  using PredicateType = typename AccessTraitsHelper<Access>::type;

  template <
      typename U = PassTag, typename V = Tag,
      std::enable_if_t<std::is_same<U, FirstPassTag>{} &&
                       std::is_same<V, SpatialPredicateTag>{}> * = nullptr>
  KOKKOS_FUNCTION auto operator()(PredicateType const &predicate,
                                  int primitive_index) const
  {
    auto const predicate_index = getData(predicate);
    auto const &raw_predicate = getPredicate(predicate);
    // With permutation, we access offset in random manner, and
    // _offset(permutated_predicate_index+1) may be in a completely different
    // place. Instead, use pointers to get the correct value for the buffer
    // size. For this reason, also take a reference for offset.
    auto const &offset = _permuted_offset(predicate_index);
    auto const buffer_size = *(&offset + 1) - offset;
    auto &count = _counts(predicate_index);

    return _callback(raw_predicate, primitive_index,
                     [&](ValueType const &value) {
                       int count_old = Kokkos::atomic_fetch_add(&count, 1);
                       if (count_old < buffer_size)
                         _out(offset + count_old) = value;
                     });
  }
  template <typename U = PassTag, typename V = Tag>
  KOKKOS_FUNCTION std::enable_if_t<std::is_same<U, FirstPassTag>{} &&
                                   std::is_same<V, NearestPredicateTag>{}>
  operator()(PredicateType const &predicate, int primitive_index,
             float distance) const
  {
    auto const predicate_index = getData(predicate);
    auto const &raw_predicate = getPredicate(predicate);
    // With permutation, we access offset in random manner, and
    // _offset(permutated_predicate_index+1) may be in a completely different
    // place. Instead, use pointers to get the correct value for the buffer
    // size. For this reason, also take a reference for offset.
    auto const &offset = _permuted_offset(predicate_index);
    auto const buffer_size = *(&offset + 1) - offset;
    auto &count = _counts(predicate_index);

    _callback(raw_predicate, primitive_index, distance,
              [&](ValueType const &value) {
                int count_old = Kokkos::atomic_fetch_add(&count, 1);
                if (count_old < buffer_size)
                  _out(offset + count_old) = value;
              });
  }

  template <
      typename U = PassTag, typename V = Tag,
      std::enable_if_t<std::is_same<U, FirstPassNoBufferOptimizationTag>{} &&
                       std::is_same<V, SpatialPredicateTag>{}> * = nullptr>
  KOKKOS_FUNCTION auto operator()(PredicateType const &predicate,
                                  int primitive_index) const
  {
    auto const predicate_index = getData(predicate);
    auto const &raw_predicate = getPredicate(predicate);

    auto &count = _counts(predicate_index);

    return _callback(raw_predicate, primitive_index, [&](ValueType const &) {
      Kokkos::atomic_fetch_add(&count, 1);
    });
  }

  template <typename U = PassTag, typename V = Tag>
  KOKKOS_FUNCTION
      std::enable_if_t<std::is_same<U, FirstPassNoBufferOptimizationTag>{} &&
                       std::is_same<V, NearestPredicateTag>{}>
      operator()(PredicateType const &predicate, int primitive_index,
                 float distance) const
  {
    auto const predicate_index = getData(predicate);
    auto const &raw_predicate = getPredicate(predicate);

    auto &count = _counts(predicate_index);

    _callback(raw_predicate, primitive_index, distance,
              [&](ValueType const &) { Kokkos::atomic_fetch_add(&count, 1); });
  }

  template <
      typename U = PassTag, typename V = Tag,
      std::enable_if_t<std::is_same<U, SecondPassTag>{} &&
                       std::is_same<V, SpatialPredicateTag>{}> * = nullptr>
  KOKKOS_FUNCTION auto operator()(PredicateType const &predicate,
                                  int primitive_index) const
  {
    auto const predicate_index = getData(predicate);
    auto const &raw_predicate = getPredicate(predicate);

    // we store offsets in counts, and offset(permute(i)) = counts(i)
    auto &offset = _counts(predicate_index);

    // TODO: there is a tradeoff here between skipping computation offset +
    // count, and atomic increment of count. I think atomically incrementing
    // offset is problematic for OpenMP as you potentially constantly steal
    // cache lines.
    return _callback(raw_predicate, primitive_index,
                     [&](ValueType const &value) {
                       _out(Kokkos::atomic_fetch_add(&offset, 1)) = value;
                     });
  }

  template <typename U = PassTag, typename V = Tag>
  KOKKOS_FUNCTION std::enable_if_t<std::is_same<U, SecondPassTag>{} &&
                                   std::is_same<V, NearestPredicateTag>{}>
  operator()(PredicateType const &predicate, int primitive_index,
             float distance) const
  {
    auto const predicate_index = getData(predicate);
    auto const &raw_predicate = getPredicate(predicate);

    // we store offsets in counts, and offset(permute(i)) = counts(i)
    auto &offset = _counts(predicate_index);

    // TODO: there is a tradeoff here between skipping computation offset +
    // count, and atomic increment of count. I think atomically incrementing
    // offset is problematic for OpenMP as you potentially constantly steal
    // cache lines.
    _callback(raw_predicate, primitive_index, distance,
              [&](ValueType const &value) {
                _out(Kokkos::atomic_fetch_add(&offset, 1)) = value;
              });
  }
};

namespace CrsGraphWrapperImpl
{

template <typename ExecutionSpace, typename Tree, typename Predicates,
          typename Callback, typename OutputView, typename OffsetView,
          typename PermuteType>
void queryImpl(ExecutionSpace const &space, Tree const &tree,
               Predicates const &predicates, Callback const &callback,
               OutputView &out, OffsetView &offset, PermuteType permute,
               BufferStatus buffer_status)
{
  // pre-condition: offset and out are preallocated. If buffer_size > 0, offset
  // is pre-initialized

  static_assert(Kokkos::is_execution_space<ExecutionSpace>{}, "");

  using Access = AccessTraits<Predicates, PredicatesTag>;
  auto const n_queries = Access::size(predicates);

  Kokkos::Profiling::pushRegion("ArborX::CrsGraphWrapper::two_pass");

  using CountView = OffsetView;
  CountView counts(Kokkos::view_alloc("ArborX::CrsGraphWrapper::counts", space),
                   n_queries);

  using PermutedPredicates =
      PermutedData<Predicates, PermuteType, true /*AttachIndices*/>;
  PermutedPredicates permuted_predicates = {predicates, permute};

  using PermutedOffset = PermutedData<OffsetView, PermuteType>;
  PermutedOffset permuted_offset = {offset, permute};

  Kokkos::Profiling::pushRegion(
      "ArborX::CrsGraphWrapper::two_pass::first_pass");
  bool underflow = false;
  bool overflow = false;
  if (buffer_status != BufferStatus::PreallocationNone)
  {
    tree.query(
        space, permuted_predicates,
        InsertGenerator<FirstPassTag, PermutedPredicates, Callback, OutputView,
                        CountView, PermutedOffset>{callback, out, counts,
                                                   permuted_offset},
        ArborX::Experimental::TraversalPolicy().setPredicateSorting(false));

    // Detecting overflow is a local operation that needs to be done for every
    // index. We allow individual buffer sizes to differ, so it's not as easy
    // as computing max counts.
    int overflow_int = 0;
    Kokkos::parallel_reduce(
        "ArborX::CrsGraphWrapper::compute_overflow",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
        KOKKOS_LAMBDA(int i, int &update) {
          auto const *const offset_ptr = &permuted_offset(i);
          if (counts(i) > *(offset_ptr + 1) - *offset_ptr)
            update = 1;
        },
        overflow_int);
    overflow = (overflow_int > 0);

    if (!overflow)
    {
      int n_results = 0;
      Kokkos::parallel_reduce(
          "ArborX::CrsGraphWrapper::compute_underflow",
          Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
          KOKKOS_LAMBDA(int i, int &update) { update += counts(i); },
          n_results);
      underflow = (n_results < out.extent_int(0));
    }
  }
  else
  {
    tree.query(
        space, permuted_predicates,
        InsertGenerator<FirstPassNoBufferOptimizationTag, PermutedPredicates,
                        Callback, OutputView, CountView, PermutedOffset>{
            callback, out, counts, permuted_offset},
        ArborX::Experimental::TraversalPolicy().setPredicateSorting(false));
    // This may not be true, but it does not matter. As long as we have
    // (n_results == 0) check before second pass, this value is not used.
    // Otherwise, we know it's overflowed as there is no allocation.
    overflow = true;
  }

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion(
      "ArborX::CrsGraphWrapper::first_pass_postprocess");

  OffsetView preallocated_offset("ArborX::CrsGraphWrapper::offset_copy", 0);
  if (underflow)
  {
    // Store a copy of the original offset. We'll need it for compression.
    preallocated_offset = clone(space, offset);
  }

  Kokkos::parallel_for(
      "ArborX::CrsGraphWrapper::copy_counts_to_offsets",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
      KOKKOS_LAMBDA(int const i) { permuted_offset(i) = counts(i); });
  exclusivePrefixSum(space, offset);

  int const n_results = lastElement(offset);

  Kokkos::Profiling::popRegion();

  if (n_results == 0)
  {
    // Exit early if either no results were found for any of the queries, or
    // nothing was inserted inside a callback for found results. This check
    // guarantees that the second pass will not be executed.
    Kokkos::resize(out, 0);
    // FIXME: do we need to reset offset if it was preallocated here?
    Kokkos::Profiling::popRegion();
    return;
  }

  if (overflow || buffer_status == BufferStatus::PreallocationNone)
  {
    // Not enough (individual) storage for results

    // If it was hard preallocation, we simply throw
    ARBORX_ASSERT(buffer_status != BufferStatus::PreallocationHard);

    // Otherwise, do the second pass
    Kokkos::Profiling::pushRegion(
        "ArborX::CrsGraphWrapper::two_pass:second_pass");

    Kokkos::parallel_for(
        "ArborX::CrsGraphWrapper::copy_offsets_to_counts",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
        KOKKOS_LAMBDA(int const i) { counts(i) = permuted_offset(i); });

    reallocWithoutInitializing(out, n_results);

    tree.query(
        space, permuted_predicates,
        InsertGenerator<SecondPassTag, PermutedPredicates, Callback, OutputView,
                        CountView, PermutedOffset>{callback, out, counts,
                                                   permuted_offset},
        ArborX::Experimental::TraversalPolicy().setPredicateSorting(false));

    Kokkos::Profiling::popRegion();
  }
  else if (underflow)
  {
    // More than enough storage for results, need compression
    Kokkos::Profiling::pushRegion(
        "ArborX::CrsGraphWrapper::two_pass:copy_values");

    OutputView tmp_out(Kokkos::ViewAllocateWithoutInitializing(out.label()),
                       n_results);

    Kokkos::parallel_for(
        "ArborX::CrsGraphWrapper::copy_valid_values",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
        KOKKOS_LAMBDA(int i) {
          int count = offset(i + 1) - offset(i);
          for (int j = 0; j < count; ++j)
          {
            tmp_out(offset(i) + j) = out(preallocated_offset(i) + j);
          }
        });
    out = tmp_out;

    Kokkos::Profiling::popRegion();
  }
  else
  {
    // The allocated storage was exactly enough for results, do nothing
  }
  Kokkos::Profiling::popRegion();
}

struct Iota
{
  KOKKOS_FUNCTION unsigned int operator()(int const i) const { return i; }
};

// Views are passed by reference here because internally Kokkos::realloc()
// is called.
template <typename Tree, typename ExecutionSpace, typename Predicates,
          typename OutputView, typename OffsetView, typename Callback>
std::enable_if_t<!is_tagged_post_callback<Callback>{} &&
                 Kokkos::is_view<OutputView>{} && Kokkos::is_view<OffsetView>{}>
queryDispatch(SpatialPredicateTag, Tree const &tree,
              ExecutionSpace const &space, Predicates const &predicates,
              Callback const &callback, OutputView &out, OffsetView &offset,
              Experimental::TraversalPolicy const &policy =
                  Experimental::TraversalPolicy())
{
  using MemorySpace = typename Tree::memory_space;
  using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;

  check_valid_callback(callback, predicates, out);

  Kokkos::Profiling::pushRegion("ArborX::CrsGraphWrapper::query::spatial");

  using Access = AccessTraits<Predicates, PredicatesTag>;
  auto const n_queries = Access::size(predicates);

  Kokkos::Profiling::pushRegion(
      "ArborX::CrsGraphWrapper::query::spatial::init_and_alloc");
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
        "ArborX::CrsGraphWrapper::query::spatial::compute_permutation");
    auto permute =
        Details::BatchedQueries<DeviceType>::sortQueriesAlongZOrderCurve(
            space, tree.bounds(), predicates);
    Kokkos::Profiling::popRegion();

    queryImpl(space, tree, predicates, callback, out, offset, permute,
              toBufferStatus(policy._buffer_size));
  }
  else
  {
    Iota permute;
    queryImpl(space, tree, predicates, callback, out, offset, permute,
              toBufferStatus(policy._buffer_size));
  }

  Kokkos::Profiling::popRegion();
}

template <typename Tree, typename ExecutionSpace, typename Predicates,
          typename Indices, typename Offset>
inline std::enable_if_t<Kokkos::is_view<Indices>{} && Kokkos::is_view<Offset>{}>
queryDispatch(SpatialPredicateTag, Tree const &tree,
              ExecutionSpace const &space, Predicates const &predicates,
              Indices &indices, Offset &offset,
              Experimental::TraversalPolicy const &policy =
                  Experimental::TraversalPolicy())
{
  queryDispatch(SpatialPredicateTag{}, tree, space, predicates,
                CallbackDefaultSpatialPredicate{}, indices, offset, policy);
}

template <typename Tree, typename ExecutionSpace, typename Predicates,
          typename OutputView, typename OffsetView, typename Callback>
inline std::enable_if_t<is_tagged_post_callback<Callback>{}>
queryDispatch(SpatialPredicateTag, Tree const &tree,
              ExecutionSpace const &space, Predicates const &predicates,
              Callback const &callback, OutputView &out, OffsetView &offset,
              Experimental::TraversalPolicy const &policy =
                  Experimental::TraversalPolicy())
{
  using MemorySpace = typename Tree::memory_space;
  Kokkos::View<int *, MemorySpace> indices(
      "ArborX::CrsGraphWrapper::query::spatial::indices", 0);
  queryDispatch(SpatialPredicateTag{}, tree, space, predicates, indices, offset,
                policy);
  callback(predicates, offset, indices, out);
}

template <typename Tree, typename ExecutionSpace, typename Predicates,
          typename OutputView, typename OffsetView, typename Callback>
std::enable_if_t<!is_tagged_post_callback<Callback>{} &&
                 Kokkos::is_view<OutputView>{} && Kokkos::is_view<OffsetView>{}>
queryDispatch(NearestPredicateTag, Tree const &tree,
              ExecutionSpace const &space, Predicates const &predicates,
              Callback const &callback, OutputView &out, OffsetView &offset,
              Experimental::TraversalPolicy const &policy =
                  Experimental::TraversalPolicy())
{
  using MemorySpace = typename Tree::memory_space;
  using DeviceType = Kokkos::Device<ExecutionSpace, MemorySpace>;

  check_valid_callback(callback, predicates, out);

  Kokkos::Profiling::pushRegion("ArborX::CrsGraphWrapper::query::nearest");

  using Access = AccessTraits<Predicates, Traits::PredicatesTag>;
  auto const n_queries = Access::size(predicates);

  Kokkos::Profiling::pushRegion(
      "ArborX::CrsGraphWrapper::query::nearest::init_and_alloc");

  reallocWithoutInitializing(offset, n_queries + 1);
  Kokkos::parallel_for(
      "ArborX::CrsGraphWrapper::query::nearest::"
      "scan_queries_for_numbers_of_nearest_neighbors",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
      KOKKOS_LAMBDA(int i) { offset(i) = getK(Access::get(predicates, i)); });
  exclusivePrefixSum(space, offset);

  int const n_results = lastElement(offset);
  reallocWithoutInitializing(out, n_results);

  Kokkos::Profiling::popRegion();

  if (policy._sort_predicates)
  {
    Kokkos::Profiling::pushRegion(
        "ArborX::CrsGraphWrapper::query::nearest::compute_permutation");
    auto permute =
        Details::BatchedQueries<DeviceType>::sortQueriesAlongZOrderCurve(
            space, tree.bounds(), predicates);
    Kokkos::Profiling::popRegion();

    queryImpl(space, tree, predicates, callback, out, offset, permute,
              BufferStatus::PreallocationSoft);
  }
  else
  {
    Iota permute;
    queryImpl(space, tree, predicates, callback, out, offset, permute,
              BufferStatus::PreallocationSoft);
  }

  Kokkos::Profiling::popRegion();
}

template <typename Tree, typename ExecutionSpace, typename Predicates,
          typename OutputView, typename OffsetView, typename Callback>
inline std::enable_if_t<is_tagged_post_callback<Callback>{}>
queryDispatch(NearestPredicateTag, Tree const &tree,
              ExecutionSpace const &space, Predicates const &predicates,
              Callback const &callback, OutputView &out, OffsetView &offset,
              Experimental::TraversalPolicy const &policy =
                  Experimental::TraversalPolicy())
{
  using MemorySpace = typename Tree::memory_space;
  Kokkos::View<Kokkos::pair<int, float> *, MemorySpace> pairs(
      "ArborX::CrsGraphWrapper::query::nearest::pairs_index_distance", 0);
  queryDispatch(NearestPredicateTag{}, tree, space, predicates,
                CallbackDefaultNearestPredicateWithDistance{}, pairs, offset,
                policy);
  callback(predicates, offset, pairs, out);
}

template <typename Tree, typename ExecutionSpace, typename Predicates,
          typename Indices, typename Offset>
inline std::enable_if_t<Kokkos::is_view<Indices>{} && Kokkos::is_view<Offset>{}>
queryDispatch(NearestPredicateTag, Tree const &tree,
              ExecutionSpace const &space, Predicates const &predicates,
              Indices &indices, Offset &offset,
              Experimental::TraversalPolicy const &policy =
                  Experimental::TraversalPolicy())
{
  queryDispatch(NearestPredicateTag{}, tree, space, predicates,
                CallbackDefaultNearestPredicate{}, indices, offset, policy);
}

template <typename Tree, typename ExecutionSpace, typename Predicates,
          typename Indices, typename Offset, typename Distances>
inline std::enable_if_t<Kokkos::is_view<Indices>{} &&
                        Kokkos::is_view<Offset>{} &&
                        Kokkos::is_view<Distances>{}>
queryDispatch(NearestPredicateTag, Tree const &tree,
              ExecutionSpace const &space, Predicates const &predicates,
              Indices &indices, Offset &offset, Distances &distances,
              Experimental::TraversalPolicy const &policy =
                  Experimental::TraversalPolicy())
{
  using MemorySpace = typename Tree::memory_space;
  Kokkos::View<Kokkos::pair<int, float> *, MemorySpace> out(
      "ArborX::CrsGraphWrapper::query::nearest::pairs_index_distance", 0);
  queryDispatch(NearestPredicateTag{}, tree, space, predicates,
                CallbackDefaultNearestPredicateWithDistance{}, out, offset,
                policy);
  auto const n = out.extent(0);
  reallocWithoutInitializing(indices, n);
  reallocWithoutInitializing(distances, n);
  Kokkos::parallel_for("ArborX::CrsGraphWrapper::query::nearest::split_pairs",
                       Kokkos::RangePolicy<ExecutionSpace>(space, 0, n),
                       KOKKOS_LAMBDA(int i) {
                         indices(i) = out(i).first;
                         distances(i) = out(i).second;
                       });
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

template <typename Tree, typename ExecutionSpace, typename Predicates,
          typename CallbackOrView, typename View, typename... Args>
inline std::enable_if_t<Kokkos::is_view<std::decay_t<View>>{}>
query(Tree const &tree, ExecutionSpace const &space,
      Predicates const &predicates, CallbackOrView &&callback_or_view,
      View &&view, Args &&... args)
{
  check_valid_callback_if_first_argument_is_not_a_view(callback_or_view,
                                                       predicates, view);

  using Access = AccessTraits<Predicates, Traits::PredicatesTag>;
  using Tag = typename AccessTraitsHelper<Access>::tag;

  queryDispatch(Tag{}, tree, space, predicates,
                std::forward<CallbackOrView>(callback_or_view),
                std::forward<View>(view), std::forward<Args>(args)...);
}

} // namespace CrsGraphWrapperImpl

} // namespace Details
} // namespace ArborX

#endif