/****************************************************************************
 * Copyright (c) 2017-2021 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_DETAILS_SORT_UTILS_HPP
#define ARBORX_DETAILS_SORT_UTILS_HPP

#include <ArborX_Config.hpp> // ARBORX_ENABLE_ROCTHRUST

#include <ArborX_DetailsUtils.hpp> // iota
#include <ArborX_Exception.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp> // min_max_functor

// clang-format off
#if defined(KOKKOS_ENABLE_CUDA)
#  if defined(KOKKOS_COMPILER_CLANG)
// Some versions of Clang fail to compile Thrust, failing with errors like
// this:
//    <snip>/thrust/system/cuda/detail/core/agent_launcher.h:557:11:
//    error: use of undeclared identifier 'va_printf'
// The exact combination of versions for Clang and Thrust (or CUDA) for this
// failure was not investigated, however even very recent version combination
// (Clang 10.0.0 and Cuda 10.0) demonstrated failure.
//
// Defining _CubLog here allows us to avoid that code path, however disabling
// some debugging diagnostics
//
// If _CubLog is already defined, we save it into ARBORX_CubLog_save, and
// restore it at the end
#    ifdef _CubLog
#      define ARBORX_CubLog_save _CubLog
#    endif
#    define _CubLog
#    include <thrust/device_ptr.h>
#    include <thrust/sort.h>
#    undef _CubLog
#    ifdef ARBORX_CubLog_save
#      define _CubLog ARBORX_CubLog_save
#      undef ARBORX_CubLog_save
#    endif
#  else // #if defined(KOKKOS_COMPILER_CLANG)
#    include <thrust/device_ptr.h>
#    include <thrust/sort.h>
#  endif // #if defined(KOKKOS_COMPILER_CLANG)
#endif   // #if defined(KOKKOS_ENABLE_CUDA)
// clang-format on

#if defined(KOKKOS_ENABLE_HIP) && defined(ARBORX_ENABLE_ROCTHRUST)
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#endif

namespace ArborX
{

namespace Details
{

template <typename ExecutionSpace, typename ValuesType>
void sort(ExecutionSpace const &space, ValuesType &values)
{
  static_assert(Kokkos::is_view<ValuesType>::value, "");
  static_assert(ValuesType::rank == 1, "");

  int const n = values.extent(0);

  using SizeType = unsigned int;
  using ValueType = typename ValuesType::value_type;
  using CompType = Kokkos::BinOp1D<ValuesType>;

  Kokkos::MinMaxScalar<ValueType> result;
  Kokkos::MinMax<ValueType> reducer(result);
  parallel_reduce("ArborX::Sorting::find_min_max_view",
                  Kokkos::RangePolicy<ExecutionSpace>(space, 0, n),
                  Kokkos::Impl::min_max_functor<ValuesType>(values), reducer);
  if (result.min_val == result.max_val)
    return;

  // FIXME Kokkos::BinSort is currently missing overloads that take an
  // execution space as argument
  Kokkos::BinSort<ValuesType, CompType, typename ValuesType::device_type,
                  SizeType>
      bin_sort(values, CompType(n / 2, result.min_val, result.max_val), true);
  bin_sort.create_permute_vector();
  bin_sort.sort(values);
}

#if defined(KOKKOS_ENABLE_CUDA) ||                                             \
    (defined(KOKKOS_ENABLE_HIP) && defined(ARBORX_ENABLE_ROCTHRUST))
template <typename ValuesType>
void sort(
#if defined(KOKKOS_ENABLE_CUDA)
    Kokkos::Cuda const &space,
#else
    Kokkos::Experimental::HIP const &space,
#endif
    ValuesType &values)
{
  static_assert(Kokkos::is_view<ValuesType>::value, "");
  static_assert(ValuesType::rank == 1, "");
  static_assert(std::is_same<std::decay_t<decltype(space)>,
                             typename ValuesType::execution_space>::value,
                "");

  int const n = values.extent(0);

#if defined(KOKKOS_ENABLE_CUDA)
  auto const execution_policy = thrust::cuda::par.on(space.cuda_stream());
#else
  auto const execution_policy = thrust::hip::par.on(space.hip_stream());
#endif

  using ValueType = typename ValuesType::value_type;

  auto values_first = thrust::device_ptr<ValueType>(values.data());
  auto values_last = thrust::device_ptr<ValueType>(values.data() + n);
  thrust::sort(execution_policy, values_first, values_last);
}
#endif

template <typename ExecutionSpace, typename KeysType, typename ValuesType>
void sortByKey(ExecutionSpace const &space, KeysType &keys, ValuesType &values)
{
  static_assert(Kokkos::is_view<KeysType>::value, "");
  static_assert(Kokkos::is_view<ValuesType>::value, "");
  static_assert(KeysType::rank == 1, "");
  static_assert(ValuesType::rank == 1, "");
  ARBORX_ASSERT(values.extent(0) == keys.extent(0));

  int const n = keys.extent(0);

  using SizeType = unsigned int;
  using ValueType = typename KeysType::value_type;
  using CompType = Kokkos::BinOp1D<KeysType>;

  Kokkos::MinMaxScalar<ValueType> result;
  Kokkos::MinMax<ValueType> reducer(result);
  parallel_reduce("ArborX::Sorting::find_min_max_view",
                  Kokkos::RangePolicy<ExecutionSpace>(space, 0, n),
                  Kokkos::Impl::min_max_functor<KeysType>(keys), reducer);
  if (result.min_val == result.max_val)
    return;

  Kokkos::BinSort<KeysType, CompType, typename KeysType::device_type, SizeType>
      bin_sort(keys, CompType(n / 2, result.min_val, result.max_val), true);
  bin_sort.create_permute_vector();
  bin_sort.sort(keys);
  // FIXME Kokkos::BinSort is currently missing overloads that take an
  // execution space as argument

  auto permute = bin_sort.get_permute_vector();
  auto values_clone = clone(values);
  Kokkos::parallel_for(
      "ArborX::Sorting::apply_permutation",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n),
      KOKKOS_LAMBDA(int const i) { values(i) = values_clone(permute(i)); });
}

#if defined(KOKKOS_ENABLE_CUDA) ||                                             \
    (defined(KOKKOS_ENABLE_HIP) && defined(ARBORX_ENABLE_ROCTHRUST))
template <typename KeysType, typename ValuesType>
void sortByKey(
#if defined(KOKKOS_ENABLE_CUDA)
    Kokkos::Cuda const &space,
#else
    Kokkos::Experimental::HIP const &space,
#endif
    KeysType &keys, ValuesType &values)
{
  static_assert(Kokkos::is_view<KeysType>::value, "");
  static_assert(Kokkos::is_view<ValuesType>::value, "");
  static_assert(KeysType::rank == 1, "");
  static_assert(ValuesType::rank == 1, "");
  static_assert(std::is_same<std::decay_t<decltype(space)>,
                             typename ValuesType::execution_space>::value,
                "");
  static_assert(std::is_same<typename ValuesType::device_type,
                             typename KeysType::device_type>::value);
  ARBORX_ASSERT(values.extent(0) == keys.extent(0));

  int const n = keys.extent(0);

#if defined(KOKKOS_ENABLE_CUDA)
  auto const execution_policy = thrust::cuda::par.on(space.cuda_stream());
#else
  auto const execution_policy = thrust::hip::par.on(space.hip_stream());
#endif

  using ValueType = typename ValuesType::value_type;
  using KeyType = typename KeysType::value_type;

  auto keys_first = thrust::device_ptr<KeyType>(keys.data());
  auto keys_last = thrust::device_ptr<KeyType>(keys.data() + n);
  auto values_first = thrust::device_ptr<ValueType>(values.data());
  thrust::sort_by_key(execution_policy, keys_first, keys_last, values_first);
}
#endif

// NOTE returns the permutation indices **and** sorts the input view
template <typename ExecutionSpace, typename ViewType,
          class SizeType = unsigned int>
Kokkos::View<SizeType *, typename ViewType::device_type>
sortObjects(ExecutionSpace const &exec_space, ViewType &view)
{
  using MemorySpace = typename ViewType::memory_space;
  Kokkos::View<SizeType *, MemorySpace> permutation_indices(
      Kokkos::ViewAllocateWithoutInitializing(
          "ArborX::Sorting::permutation_indices"),
      view.size());
  iota(exec_space, permutation_indices);
  Details::sortByKey(exec_space, view, permutation_indices);
  return permutation_indices;
}

// Helper functions and structs for applyPermutations
namespace PermuteHelper
{
template <class DstViewType, class SrcViewType, int Rank = DstViewType::Rank>
struct CopyOp;

template <class DstViewType, class SrcViewType>
struct CopyOp<DstViewType, SrcViewType, 1>
{
  KOKKOS_INLINE_FUNCTION
  static void copy(DstViewType const &dst, size_t i_dst, SrcViewType const &src,
                   size_t i_src)
  {
    dst(i_dst) = src(i_src);
  }
};

template <class DstViewType, class SrcViewType>
struct CopyOp<DstViewType, SrcViewType, 2>
{
  KOKKOS_INLINE_FUNCTION
  static void copy(DstViewType const &dst, size_t i_dst, SrcViewType const &src,
                   size_t i_src)
  {
    for (unsigned int j = 0; j < dst.extent(1); j++)
      dst(i_dst, j) = src(i_src, j);
  }
};

template <class DstViewType, class SrcViewType>
struct CopyOp<DstViewType, SrcViewType, 3>
{
  KOKKOS_INLINE_FUNCTION
  static void copy(DstViewType const &dst, size_t i_dst, SrcViewType const &src,
                   size_t i_src)
  {
    for (unsigned int j = 0; j < dst.extent(1); j++)
      for (unsigned int k = 0; k < dst.extent(2); k++)
        dst(i_dst, j, k) = src(i_src, j, k);
  }
};
} // namespace PermuteHelper

template <typename ExecutionSpace, typename PermutationView, typename InputView,
          typename OutputView>
void applyInversePermutation(ExecutionSpace const &space,
                             PermutationView const &permutation,
                             InputView const &input_view,
                             OutputView const &output_view)
{
  static_assert(std::is_integral<typename PermutationView::value_type>::value,
                "");
  ARBORX_ASSERT(permutation.extent(0) == input_view.extent(0));
  ARBORX_ASSERT(output_view.extent(0) == input_view.extent(0));

  Kokkos::parallel_for(
      "ArborX::Sorting::inverse_permute",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, input_view.extent(0)),
      KOKKOS_LAMBDA(int i) {
        PermuteHelper::CopyOp<OutputView, InputView>::copy(
            output_view, permutation(i), input_view, i);
      });
}

template <typename ExecutionSpace, typename PermutationView, typename InputView,
          typename OutputView>
void applyPermutation(ExecutionSpace const &space,
                      PermutationView const &permutation,
                      InputView const &input_view,
                      OutputView const &output_view)
{
  static_assert(std::is_integral<typename PermutationView::value_type>::value,
                "");
  ARBORX_ASSERT(permutation.extent(0) == input_view.extent(0));
  ARBORX_ASSERT(output_view.extent(0) == input_view.extent(0));

  Kokkos::parallel_for(
      "ArborX::Sorting::permute",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, input_view.extent(0)),
      KOKKOS_LAMBDA(int i) {
        PermuteHelper::CopyOp<OutputView, InputView>::copy(
            output_view, i, input_view, permutation(i));
      });
}

template <typename ExecutionSpace, typename PermutationView, typename View>
void applyPermutation(ExecutionSpace const &space,
                      PermutationView const &permutation, View &view)
{
  static_assert(std::is_integral<typename PermutationView::value_type>::value,
                "");
  auto scratch_view = clone(space, view);
  applyPermutation(space, permutation, scratch_view, view);
}

} // namespace Details

} // namespace ArborX

#endif
