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
#ifndef ARBORX_DETAILS_BUFFER_OPTIMIZATON_HPP
#define ARBORX_DETAILS_BUFFER_OPTIMIZATON_HPP

#include <ArborX_AccessTraits.hpp>
#include <ArborX_DetailsUtils.hpp>
#include <ArborX_Exception.hpp>

#include <Kokkos_Core.hpp>

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

template <typename Data, typename Permute, bool AttachIndices = false>
struct PermutedData
{
  Data _data;
  Permute _permute;
  KOKKOS_FUNCTION auto &operator()(int i) const { return _data(_permute(i)); }
};

} // namespace Details

template <typename Predicates, typename Permute, bool AttachIndices>
struct AccessTraits<Details::PermutedData<Predicates, Permute, AttachIndices>,
                    PredicatesTag>
{
  using PermutedPredicates =
      Details::PermutedData<Predicates, Permute, AttachIndices>;
  using NativeAccess = AccessTraits<Predicates, PredicatesTag>;

  static std::size_t size(PermutedPredicates const &permuted_predicates)
  {
    return NativeAccess::size(permuted_predicates._data);
  }

  template <bool _Attach = AttachIndices>
  KOKKOS_FUNCTION static auto get(PermutedPredicates const &permuted_predicates,
                                  std::enable_if_t<_Attach, std::size_t> index)
  {
    auto const permuted_index = permuted_predicates._permute(index);
    return attach(NativeAccess::get(permuted_predicates._data, permuted_index),
                  (int)index);
  }

  template <bool _Attach = AttachIndices>
  KOKKOS_FUNCTION static auto get(PermutedPredicates const &permuted_predicates,
                                  std::enable_if_t<!_Attach, std::size_t> index)
  {
    auto const permuted_index = permuted_predicates._permute(index);
    return NativeAccess::get(permuted_predicates._data, permuted_index);
  }
  using memory_space = typename NativeAccess::memory_space;
};

namespace Details
{

template <typename ExecutionSpace, typename TreeTraversal, typename Predicates,
          typename Callback, typename OutputView, typename OffsetView,
          typename PermuteType>
void queryImpl(ExecutionSpace const &space, TreeTraversal const &tree_traversal,
               Predicates const &predicates, Callback const &callback,
               OutputView &out, OffsetView &offset, PermuteType permute,
               BufferStatus buffer_status)
{
  // pre-condition: offset and out are preallocated. If buffer_size > 0, offset
  // is pre-initialized

  static_assert(Kokkos::is_execution_space<ExecutionSpace>{}, "");

  using Access = AccessTraits<Predicates, PredicatesTag>;
  auto const n_queries = Access::size(predicates);

  Kokkos::Profiling::pushRegion("ArborX::BufferOptimization::two_pass");

  using CountView = OffsetView;
  CountView counts(
      Kokkos::view_alloc("ArborX::BufferOptimization::counts", space),
      n_queries);

  using PermutedPredicates =
      PermutedData<Predicates, PermuteType, true /*AttachIndices*/>;
  PermutedPredicates permuted_predicates = {predicates, permute};

  using PermutedOffset = PermutedData<OffsetView, PermuteType>;
  PermutedOffset permuted_offset = {offset, permute};

  Kokkos::Profiling::pushRegion(
      "ArborX::BufferOptimization::two_pass::first_pass");
  bool underflow = false;
  bool overflow = false;
  if (buffer_status != BufferStatus::PreallocationNone)
  {
    tree_traversal.launch(
        space, permuted_predicates,
        InsertGenerator<FirstPassTag, PermutedPredicates, Callback, OutputView,
                        CountView, PermutedOffset>{callback, out, counts,
                                                   permuted_offset});

    // Detecting overflow is a local operation that needs to be done for every
    // index. We allow individual buffer sizes to differ, so it's not as easy
    // as computing max counts.
    int overflow_int = 0;
    Kokkos::parallel_reduce(
        "ArborX::BufferOptimization::compute_overflow",
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
          "ArborX::BufferOptimization::compute_underflow",
          Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
          KOKKOS_LAMBDA(int i, int &update) { update += counts(i); },
          n_results);
      underflow = (n_results < out.extent_int(0));
    }
  }
  else
  {
    tree_traversal.launch(
        space, permuted_predicates,
        InsertGenerator<FirstPassNoBufferOptimizationTag, PermutedPredicates,
                        Callback, OutputView, CountView, PermutedOffset>{
            callback, out, counts, permuted_offset});
    // This may not be true, but it does not matter. As long as we have
    // (n_results == 0) check before second pass, this value is not used.
    // Otherwise, we know it's overflowed as there is no allocation.
    overflow = true;
  }

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion(
      "ArborX::BufferOptimization::first_pass_postprocess");

  OffsetView preallocated_offset("ArborX::BufferOptimization::offset_copy", 0);
  if (underflow)
  {
    // Store a copy of the original offset. We'll need it for compression.
    preallocated_offset = clone(space, offset);
  }

  Kokkos::parallel_for(
      "ArborX::BufferOptimization::copy_counts_to_offsets",
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
        "ArborX::BufferOptimization::two_pass:second_pass");

    Kokkos::parallel_for(
        "ArborX::BufferOptimization::copy_offsets_to_counts",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
        KOKKOS_LAMBDA(int const i) { counts(i) = permuted_offset(i); });

    reallocWithoutInitializing(out, n_results);

    tree_traversal.launch(
        space, permuted_predicates,
        InsertGenerator<SecondPassTag, PermutedPredicates, Callback, OutputView,
                        CountView, PermutedOffset>{callback, out, counts,
                                                   permuted_offset});

    Kokkos::Profiling::popRegion();
  }
  else if (underflow)
  {
    // More than enough storage for results, need compression
    Kokkos::Profiling::pushRegion(
        "ArborX::BufferOptimization::two_pass:copy_values");

    OutputView tmp_out(Kokkos::ViewAllocateWithoutInitializing(out.label()),
                       n_results);

    Kokkos::parallel_for(
        "ArborX::BufferOptimization::copy_valid_values",
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

} // namespace Details
} // namespace ArborX

#endif
