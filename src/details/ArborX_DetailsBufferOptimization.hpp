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

template <typename Predicates, typename CountView, typename OffsetView>
struct PredicatesWithCountsAndOffsets
{
  using predicates_type = Predicates;

  Predicates _predicates;
  CountView _counts;
  OffsetView _offset;
};

template <typename Predicates, typename CountView>
struct PredicatesWithCounts
{
  using predicates_type = Predicates;

  Predicates _predicates;
  CountView _counts;
};

} // namespace Details

namespace Traits
{
template <typename Predicates, typename CountView, typename OffsetView>
struct Access<
    Details::PredicatesWithCountsAndOffsets<Predicates, CountView, OffsetView>,
    PredicatesTag>
{
  using WrappedPredicates =
      Details::PredicatesWithCountsAndOffsets<Predicates, CountView,
                                              OffsetView>;

  using NativeAccess = Access<Predicates, PredicatesTag>;

  inline static std::size_t size(WrappedPredicates const &wrapped_predicates)
  {
    return NativeAccess::size(wrapped_predicates._predicates);
  }

  KOKKOS_INLINE_FUNCTION static auto
  get(WrappedPredicates const &wrapped_predicates, std::size_t i)
  {
    return attach(NativeAccess::get(wrapped_predicates._predicates, i),
                  Kokkos::make_pair(wrapped_predicates._counts.data() + i,
                                    wrapped_predicates._offset.data() + i));
  }
  using memory_space = typename NativeAccess::memory_space;
};
template <typename Predicates, typename CountView>
struct Access<Details::PredicatesWithCounts<Predicates, CountView>,
              PredicatesTag>
{
  using WrappedPredicates =
      Details::PredicatesWithCounts<Predicates, CountView>;

  using NativeAccess = Access<Predicates, PredicatesTag>;

  inline static std::size_t size(WrappedPredicates const &wrapped_predicates)
  {
    return NativeAccess::size(wrapped_predicates._predicates);
  }

  KOKKOS_INLINE_FUNCTION static auto
  get(WrappedPredicates const &wrapped_predicates, std::size_t i)
  {
    return attach(NativeAccess::get(wrapped_predicates._predicates, i),
                  wrapped_predicates._counts.data() + i);
  }
  using memory_space = typename NativeAccess::memory_space;
};
} // namespace Traits

namespace Details
{

template <typename Tag, typename WrappedPredicatesType, typename Callback,
          typename OutputView>
struct InsertGenerator
{
  using Predicates = typename WrappedPredicatesType::predicates_type;
  Predicates _predicates;
  WrappedPredicatesType _wrapped_predicates;
  Callback _callback;
  OutputView _out;

  using ValueType = typename OutputView::value_type;
  using Access = Traits::Access<Predicates, Traits::PredicatesTag>;
  using WrappedAccess =
      Traits::Access<WrappedPredicatesType, Traits::PredicatesTag>;

  template <typename U = Tag>
  KOKKOS_FUNCTION std::enable_if_t<std::is_same<U, FirstPassTag>::value>
  operator()(int predicate_index, int primitive_index) const
  {
    int i = primitive_index;
    int j = predicate_index;

    auto data = getData(WrappedAccess::get(_wrapped_predicates, j));
    auto *count_ptr = data.first;
    auto *offset_ptr = data.second;

    auto offset = *offset_ptr;
    auto offset_next = *(offset_ptr + 1);

    // FIXME: race condition here, counts could be read but then incremented by
    // a different thread
    if (offset + *count_ptr < offset_next)
      _callback(Access::get(_predicates, j), i, [&](ValueType const &value) {
        _out(offset + Kokkos::atomic_fetch_add(count_ptr, 1)) = value;
      });
    else
      _callback(Access::get(_predicates, j), i, [&](ValueType const &) {
        Kokkos::atomic_fetch_add(count_ptr, 1);
      });
  }

  template <typename U = Tag>
  KOKKOS_FUNCTION
      std::enable_if_t<std::is_same<U, FirstPassNoBufferOptimizationTag>::value>
      operator()(int predicate_index, int primitive_index) const
  {
    int i = primitive_index;
    int j = predicate_index;

    auto *count_ptr = getData(WrappedAccess::get(_wrapped_predicates, j));

    _callback(Access::get(_predicates, j), i, [&](ValueType const &) {
      Kokkos::atomic_fetch_add(count_ptr, 1);
    });
  }

  template <typename U = Tag>
  KOKKOS_FUNCTION std::enable_if_t<std::is_same<U, SecondPassTag>::value>
  operator()(int predicate_index, int primitive_index) const
  {
    int i = primitive_index;
    int j = predicate_index;

    auto *offset_ptr = getData(WrappedAccess::get(_wrapped_predicates, j));

    _callback(Access::get(_predicates, j), i, [&](ValueType const &value) {
      _out(Kokkos::atomic_fetch_add(offset_ptr, 1)) = value;
    });
  }
};

template <typename ExecutionSpace, typename TreeTraversal, typename Predicates,
          typename Callback, typename OutputView, typename OffsetView>
void spatialQueryImpl(ExecutionSpace const &space,
                      TreeTraversal const &tree_traversal,
                      Predicates const &predicates, Callback const &callback,
                      OutputView &out, OffsetView &offset, int buffer_size)
{
  static_assert(Kokkos::is_execution_space<ExecutionSpace>{}, "");

  using Access = Traits::Access<Predicates, Traits::PredicatesTag>;
  auto const n_queries = Access::size(predicates);

  bool const throw_if_buffer_optimization_fails = (buffer_size < 0);
  buffer_size = std::abs(buffer_size);

  reallocWithoutInitializing(offset, n_queries + 1);

  using CountView = OffsetView;
  CountView counts(Kokkos::view_alloc("counts", space), n_queries + 1);

  using Predicates1 = PredicatesWithCounts<Predicates, CountView>;
  using Predicates2 =
      PredicatesWithCountsAndOffsets<Predicates, CountView, OffsetView>;
  Predicates1 predicates1 = {predicates, counts};
  Predicates2 predicates2 = {predicates, counts, offset};

  if (buffer_size > 0)
  {
    Kokkos::deep_copy(space, offset, buffer_size); // FIXME
    exclusivePrefixSum(space, offset);             // FIXME

    reallocWithoutInitializing(out, n_queries * buffer_size);
    // NOTE I considered filling with invalid indices but it is unecessary work

    tree_traversal.launch(
        space, predicates2,
        InsertGenerator<FirstPassTag, Predicates2, Callback, OutputView>{
            predicates, predicates2, callback, out});
  }
  else
  {
    tree_traversal.launch(
        space, predicates1,
        InsertGenerator<FirstPassNoBufferOptimizationTag, Predicates1, Callback,
                        OutputView>{predicates, predicates1, callback, out});
  }

  // NOTE max() internally calls Kokkos::parallel_reduce.  Only pay for it if
  // actually trying buffer optimization. In principle, any strictly
  // positive value can be assigned otherwise.
  auto const max_results_per_query =
      (buffer_size > 0)
          ? max(space, counts)
          : std::numeric_limits<typename std::remove_reference<decltype(
                offset)>::type::value_type>::max();

  exclusivePrefixSum(space, counts);
  Kokkos::deep_copy(space, offset, counts);

  int const n_results = lastElement(offset);

  // Exit early if either no results were found for any of the queries, or
  // nothing was inserted inside a callback for found results. This check
  // guarantees that the second pass will not be executed independent of
  // buffer_size.
  if (n_results == 0)
    return;

  if (max_results_per_query > buffer_size)
  {
    // FIXME can definitely do better about error message
    ARBORX_ASSERT(!throw_if_buffer_optimization_fails);

    reallocWithoutInitializing(out, n_results);

    tree_traversal.launch(
        space, predicates1,
        InsertGenerator<SecondPassTag, Predicates1, Callback, OutputView>{
            predicates, predicates1, callback, out});
  }
  // do not copy if by some miracle each query exactly yielded as many results
  // as the buffer size
  else if (n_results != static_cast<int>(n_queries) * buffer_size)
  {
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
  }
}

} // namespace Details
} // namespace ArborX

#endif
