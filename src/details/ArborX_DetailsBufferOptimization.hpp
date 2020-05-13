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
#include <ArborX_Macros.hpp>

#include <Kokkos_Core.hpp>

namespace ArborX
{
namespace Details
{

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
          typename OutputView, typename CountView, typename OffsetView,
          typename PermuteType>
struct InsertGenerator
{
  Predicates _permuted_predicates;
  Callback _callback;
  OutputView _out;
  CountView _counts;
  OffsetView _offset;
  PermuteType _permute;

  using ValueType = typename OutputView::value_type;
  using Access = Traits::Access<Predicates, Traits::PredicatesTag>;
  using Tag = typename Traits::Helper<Access>::tag;

  template <typename U = PassTag, typename V = Tag>
  KOKKOS_FUNCTION std::enable_if_t<std::is_same<U, FirstPassTag>{} &&
                                   std::is_same<V, SpatialPredicateTag>{}>
  operator()(int predicate_index, int primitive_index) const
  {
    auto const permuted_predicate_index = _permute(predicate_index);
    auto const offset = _offset(permuted_predicate_index);
    // With permutation, we access offset in random manner, and
    // _offset(permutated_predicate_index+1) may be in a completely different
    // place. Instead, use pointers to get the correct value for the buffer
    // size.
    auto const buffer_size = *(&offset + 1) - offset;
    auto &count = _counts(predicate_index);

    _callback(Access::get(_permuted_predicates, predicate_index),
              primitive_index, [&](ValueType const &value) {
                int count_old = Kokkos::atomic_fetch_add(&count, 1);
                if (count_old < buffer_size)
                  _out(offset + count_old) = value;
              });
  }
  template <typename U = PassTag, typename V = Tag>
  KOKKOS_FUNCTION std::enable_if_t<std::is_same<U, FirstPassTag>{} &&
                                   std::is_same<V, NearestPredicateTag>{}>
  operator()(int predicate_index, int primitive_index, float distance) const
  {
    auto const permuted_predicate_index = _permute(predicate_index);
    auto const offset = _offset(permuted_predicate_index);
    // With permutation, we access offset in random manner, and
    // _offset(permutated_predicate_index+1) may be in a completely different
    // place. Instead, use pointers to get the correct value for the buffer
    // size.
    auto const buffer_size = *(&offset + 1) - offset;
    auto &count = _counts(predicate_index);

    _callback(Access::get(_permuted_predicates, predicate_index),
              primitive_index, distance, [&](ValueType const &value) {
                int count_old = Kokkos::atomic_fetch_add(&count, 1);
                if (count_old < buffer_size)
                  _out(offset + count_old) = value;
              });
  }

  template <typename U = PassTag, typename V = Tag>
  KOKKOS_FUNCTION
      std::enable_if_t<std::is_same<U, FirstPassNoBufferOptimizationTag>{} &&
                       std::is_same<V, SpatialPredicateTag>{}>
      operator()(int predicate_index, int primitive_index) const
  {
    auto &count = _counts(predicate_index);

    _callback(Access::get(_permuted_predicates, predicate_index),
              primitive_index,
              [&](ValueType const &) { Kokkos::atomic_fetch_add(&count, 1); });
  }

  template <typename U = PassTag, typename V = Tag>
  KOKKOS_FUNCTION
      std::enable_if_t<std::is_same<U, FirstPassNoBufferOptimizationTag>{} &&
                       std::is_same<V, NearestPredicateTag>{}>
      operator()(int predicate_index, int primitive_index, float distance) const
  {
    auto &count = _counts(predicate_index);

    _callback(Access::get(_permuted_predicates, predicate_index),
              primitive_index, distance,
              [&](ValueType const &) { Kokkos::atomic_fetch_add(&count, 1); });
  }

  template <typename U = PassTag, typename V = Tag>
  KOKKOS_FUNCTION std::enable_if_t<std::is_same<U, SecondPassTag>{} &&
                                   std::is_same<V, SpatialPredicateTag>{}>
  operator()(int predicate_index, int primitive_index) const
  {
    // we store offsets in counts, and offset(permute(i)) = counts(i)
    auto &offset = _counts(predicate_index);

    _callback(Access::get(_permuted_predicates, predicate_index),
              primitive_index, [&](ValueType const &value) {
                _out(Kokkos::atomic_fetch_add(&offset, 1)) = value;
              });
  }

  template <typename U = PassTag, typename V = Tag>
  KOKKOS_FUNCTION std::enable_if_t<std::is_same<U, SecondPassTag>{} &&
                                   std::is_same<V, NearestPredicateTag>{}>
  operator()(int predicate_index, int primitive_index, float distance) const
  {
    // we store offsets in counts, and offset(permute(i)) = counts(i)
    auto &offset = _counts(predicate_index);

    _callback(Access::get(_permuted_predicates, predicate_index),
              primitive_index, distance, [&](ValueType const &value) {
                _out(Kokkos::atomic_fetch_add(&offset, 1)) = value;
              });
  }
};

template <typename Predicates, typename Permute>
struct PermutedPredicates
{
  Predicates _predicates;
  Permute _permute;
  KOKKOS_FUNCTION auto operator()(int i) const
  {
    return _predicates(_permute(i));
  }
};

} // namespace Details

namespace Traits
{
template <typename Predicates, typename Permute>
struct Access<Details::PermutedPredicates<Predicates, Permute>, PredicatesTag>
{
  using PermutedPredicates = Details::PermutedPredicates<Predicates, Permute>;
  using NativeAccess = Access<Predicates, PredicatesTag>;

  inline static std::size_t size(PermutedPredicates const &permuted_predicates)
  {
    return NativeAccess::size(permuted_predicates._predicates);
  }

  KOKKOS_INLINE_FUNCTION static auto
  get(PermutedPredicates const &permuted_predicates, std::size_t i)
  {
    return NativeAccess::get(permuted_predicates._predicates,
                             permuted_predicates._permute(i));
  }
  using memory_space = typename NativeAccess::memory_space;
};
} // namespace Traits

namespace Details
{
template <typename ExecutionSpace, typename TreeTraversal, typename Predicates,
          typename Callback, typename OutputView, typename OffsetView,
          typename PermuteType>
void queryImpl(ExecutionSpace const &space, TreeTraversal const &tree_traversal,
               Predicates const &predicates, Callback const &callback,
               OutputView &out, OffsetView &offset, PermuteType permute,
               int buffer_size)
{
  // pre-condition: offset and out are preallocated. If buffer_size > 0, offset
  // is pre-initialized

  static_assert(Kokkos::is_execution_space<ExecutionSpace>{}, "");

  using Access = Traits::Access<Predicates, Traits::PredicatesTag>;
  auto const n_queries = Access::size(predicates);

  Kokkos::Profiling::pushRegion("ArborX:BVH:two_pass");

  bool const throw_if_buffer_optimization_fails = (buffer_size < 0);
  buffer_size = std::abs(buffer_size);

  using CountView = OffsetView;
  CountView counts(Kokkos::view_alloc("counts", space), n_queries);

  using PermutedPredicates = PermutedPredicates<Predicates, PermuteType>;
  PermutedPredicates permuted_predicates = {predicates, permute};

  Kokkos::Profiling::pushRegion("ArborX:BVH:two_pass:first_pass");
  if (buffer_size > 0)
  {
    tree_traversal.launch(
        space, permuted_predicates,
        InsertGenerator<FirstPassTag, PermutedPredicates, Callback, OutputView,
                        CountView, OffsetView, PermuteType>{
            permuted_predicates, callback, out, counts, offset, permute});
  }
  else
  {
    tree_traversal.launch(
        space, permuted_predicates,
        InsertGenerator<FirstPassNoBufferOptimizationTag, PermutedPredicates,
                        Callback, OutputView, CountView, OffsetView,
                        PermuteType>{permuted_predicates, callback, out, counts,
                                     offset, permute});
  }

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX:BVH:two_pass:first_pass_postprocess");

  // FIXME
  int num_bad_guesses = 0;
  if (buffer_size > 0)
  {
    Kokkos::parallel_reduce(
        ARBORX_MARK_REGION("check_if_second_pass_needed"),
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
        KOKKOS_LAMBDA(int i, int &update) {
          auto const *const offset_ptr = &offset(permute(i));
          if (counts(i) > *(offset_ptr + 1) - *offset_ptr)
          {
            ++update;
          }
        },
        num_bad_guesses);
  }

  Kokkos::parallel_for(
      ARBORX_MARK_REGION("copy_counts_to_offsets"),
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
      KOKKOS_LAMBDA(int const i) { offset(permute(i)) = counts(i); });
  exclusivePrefixSum(space, offset);

  int const n_results = lastElement(offset);

  Kokkos::Profiling::popRegion();

  // Exit early if either no results were found for any of the queries, or
  // nothing was inserted inside a callback for found results. This check
  // guarantees that the second pass will not be executed independent of
  // buffer_size.
  if (n_results == 0)
  {
    Kokkos::resize(out, 0);
    Kokkos::Profiling::popRegion();
    return;
  }

  if (num_bad_guesses > 0 || buffer_size == 0)
  {
    Kokkos::Profiling::pushRegion("ArborX:BVH:two_pass:second_pass");

    // FIXME can definitely do better about error message
    ARBORX_ASSERT(!throw_if_buffer_optimization_fails);

    Kokkos::parallel_for(
        ARBORX_MARK_REGION("copy_offsets_to_counts"),
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
        KOKKOS_LAMBDA(int const i) { counts(i) = offset(permute(i)); });

    reallocWithoutInitializing(out, n_results);

    tree_traversal.launch(
        space, permuted_predicates,
        InsertGenerator<SecondPassTag, PermutedPredicates, Callback, OutputView,
                        CountView, OffsetView, PermuteType>{
            permuted_predicates, callback, out, counts, offset, permute});

    Kokkos::Profiling::popRegion();
  }
  // do not copy if by some miracle each query exactly yielded as many results
  // as the buffer size
  else if (n_results != static_cast<int>(n_queries) * buffer_size)
  {
    Kokkos::Profiling::pushRegion("ArborX:BVH:two_pass:copy_values");

    OutputView tmp_out(Kokkos::ViewAllocateWithoutInitializing(out.label()),
                       n_results);

    Kokkos::parallel_for(
        ARBORX_MARK_REGION("copy_valid_values"),
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
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
} // namespace Details

} // namespace Details
} // namespace ArborX

#endif
