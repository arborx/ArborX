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

#include <ArborX_Exception.hpp>
#include <ArborX_Macros.hpp>
#include <ArborX_Traits.hpp>

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

template <typename Tag, typename Predicates, typename Callback,
          typename OutputView, typename CountView, typename OffsetView>
struct WrappedCallback;

template <typename Predicates, typename Callback, typename OutputView,
          typename CountView, typename OffsetView>
struct WrappedCallback<FirstPassTag, Predicates, Callback, OutputView,
                       CountView, OffsetView>
{
  Predicates predicates_;
  Callback callback_;
  OutputView out_;
  CountView counts_;
  OffsetView offset_;
  int j;

  using ValueType = typename OutputView::value_type;
  using Access = Traits::Access<Predicates, Traits::PredicatesTag>;

  KOKKOS_FUNCTION void operator()(int i) const
  {
    if (offset_(j) + counts_(j) < offset_(j + 1))
      callback_(Access::get(predicates_, j), i, [&](ValueType const &value) {
        out_(offset_(j) + Kokkos::atomic_fetch_add(&counts_(j), 1)) = value;
      });
    else
      callback_(Access::get(predicates_, j), i, [&](ValueType const &) {
        Kokkos::atomic_fetch_add(&counts_(j), 1);
      });
  }
};

template <typename Predicates, typename Callback, typename OutputView,
          typename CountView, typename OffsetView>
struct WrappedCallback<FirstPassNoBufferOptimizationTag, Predicates, Callback,
                       OutputView, CountView, OffsetView>
{
  Predicates predicates_;
  Callback callback_;
  OutputView out_;
  CountView counts_;
  OffsetView offset_;
  int j;

  using ValueType = typename OutputView::value_type;
  using Access = Traits::Access<Predicates, Traits::PredicatesTag>;

  KOKKOS_FUNCTION void operator()(int i) const
  {
    callback_(Access::get(predicates_, j), i, [&](ValueType const &) {
      Kokkos::atomic_fetch_add(&counts_(j), 1);
    });
  }
};

template <typename Predicates, typename Callback, typename OutputView,
          typename CountView, typename OffsetView>
struct WrappedCallback<SecondPassTag, Predicates, Callback, OutputView,
                       CountView, OffsetView>
{
  Predicates predicates_;
  Callback callback_;
  OutputView out_;
  CountView counts_;
  OffsetView offset_;
  int j;

  using ValueType = typename OutputView::value_type;
  using Access = Traits::Access<Predicates, Traits::PredicatesTag>;

  KOKKOS_FUNCTION void operator()(int i) const
  {
    callback_(Access::get(predicates_, j), i, [&](ValueType const &value) {
      out_(offset_(j) + Kokkos::atomic_fetch_add(&counts_(j), 1)) = value;
    });
  }
};

template <typename Tag, typename Predicates, typename Callback,
          typename OutputView, typename CountView, typename OffsetView>
struct CallbackGenerator
{
  Predicates predicates_;
  Callback callback_;
  OutputView out_;
  CountView counts_;
  OffsetView offset_;

  KOKKOS_FUNCTION WrappedCallback<Tag, Predicates, Callback, OutputView,
                                  CountView, OffsetView>
  operator()(int j) const
  {
    return {predicates_, callback_, out_, counts_, offset_, j};
  }
};

template <typename ExecutionSpace, typename Search, typename Predicates,
          typename Callback, typename OutputView, typename OffsetView>
void queryImpl(ExecutionSpace const &space, Search const &search,
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
  CountView counts(Kokkos::view_alloc("counts", space), n_queries);

  if (buffer_size > 0)
  {
    Kokkos::deep_copy(space, offset, buffer_size); // FIXME
    exclusivePrefixSum(space, offset);             // FIXME

    reallocWithoutInitializing(out, n_queries * buffer_size);
    // NOTE I considered filling with invalid indices but it is unecessary work

    search(space, predicates,
           CallbackGenerator<FirstPassTag, Predicates, Callback, OutputView,
                             CountView, OffsetView>{predicates, callback, out,
                                                    counts, offset});
  }
  else
  {
    search(space, predicates,
           CallbackGenerator<FirstPassNoBufferOptimizationTag, Predicates,
                             Callback, OutputView, CountView, OffsetView>{
               predicates, callback, out, counts, offset});
  }

  // NOTE max() internally calls Kokkos::parallel_reduce.  Only pay for it if
  // actually trying buffer optimization. In principle, any strictly
  // positive value can be assigned otherwise.
  auto const max_results_per_query =
      (buffer_size > 0)
          ? max(space, counts)
          : std::numeric_limits<typename std::remove_reference<decltype(
                offset)>::type::value_type>::max();

  Kokkos::deep_copy(
      space, subview(offset, Kokkos::make_pair(0, (int)n_queries)), counts);
  exclusivePrefixSum(space, offset);

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

    Kokkos::deep_copy(space, counts, 0); // FIXME

    search(space, predicates,
           CallbackGenerator<SecondPassTag, Predicates, Callback, OutputView,
                             CountView, OffsetView>{predicates, callback, out,
                                                    counts, offset});
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
