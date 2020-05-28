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
#include <Kokkos_UnorderedMap.hpp>

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

BufferStatus toBufferStatus(int buffer_size)
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
          typename OutputView, typename CountView, typename OffsetView,
          typename PermuteType>
struct InsertGenerator
{
  Predicates _permuted_predicates;
  Callback _callback;
  OutputView _out;
  CountView _counts;

  using ValueType = typename OutputView::value_type;
  using Access = Traits::Access<Predicates, Traits::PredicatesTag>;
  using Tag = typename Traits::Helper<Access>::tag;

  template <typename U = PassTag, typename V = Tag>
  KOKKOS_FUNCTION std::enable_if_t<std::is_same<U, FirstPassTag>{} &&
                                   std::is_same<V, SpatialPredicateTag>{}>
  operator()(int predicate_index, int primitive_index) const
  {
    auto &count = _counts(predicate_index);

    _callback(Access::get(_permuted_predicates, predicate_index),
              primitive_index, [&](ValueType const &value) {
                int count_old = Kokkos::atomic_fetch_add(&count, 1);
                _out.insert(Kokkos::pair<int, int>{predicate_index, count_old},
                            value);
              });
  }
  template <typename U = PassTag, typename V = Tag>
  KOKKOS_FUNCTION std::enable_if_t<std::is_same<U, FirstPassTag>{} &&
                                   std::is_same<V, NearestPredicateTag>{}>
  operator()(int predicate_index, int primitive_index, float distance) const
  {
    auto &count = _counts(predicate_index);

    _callback(Access::get(_permuted_predicates, predicate_index),
              primitive_index, distance, [&](ValueType const &value) {
                int count_old = Kokkos::atomic_fetch_add(&count, 1);
                _out.insert(Kokkos::pair<int, int>{predicate_index, count_old},
                            value);
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
               BufferStatus buffer_status)
{
  // pre-condition: offset and out are preallocated. If buffer_size > 0, offset
  // is pre-initialized

  using MapType =
      Kokkos::UnorderedMap<Kokkos::pair<int, int>,
                           typename OutputView::value_type, ExecutionSpace>;
  MapType unordered_map(1000000);

  static_assert(Kokkos::is_execution_space<ExecutionSpace>{}, "");

  using Access = Traits::Access<Predicates, Traits::PredicatesTag>;
  auto const n_queries = Access::size(predicates);

  using CountView = Kokkos::View<int *, ExecutionSpace>;
  CountView counts(Kokkos::view_alloc("counts", space), n_queries);

  using PermutedPredicates = PermutedPredicates<Predicates, PermuteType>;
  PermutedPredicates permuted_predicates = {predicates, permute};

  tree_traversal.launch(
      space, permuted_predicates,
      InsertGenerator<FirstPassTag, PermutedPredicates, Callback, MapType,
                      CountView, OffsetView, PermuteType>{
          permuted_predicates, callback, unordered_map, counts});

  Kokkos::parallel_for(
      ARBORX_MARK_REGION("copy_counts_to_offsets"),
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_queries),
      KOKKOS_LAMBDA(int const i) { offset(permute(i)) = counts(i); });
  exclusivePrefixSum(space, offset);

  int const n_results = lastElement(offset);
  reallocWithoutInitializing(out, n_results);

  // fill the output view from the unordered_map
  Kokkos::parallel_for(
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, unordered_map.capacity()),
      KOKKOS_LAMBDA(uint32_t i) {
        if (unordered_map.valid_at(i))
        {
          auto key = unordered_map.key_at(i);
          auto value = unordered_map.value_at(i);
          out(offset(permute(key.first)) + key.second) = value;
        }
      });
} // namespace Details

} // namespace Details
} // namespace ArborX

#endif
