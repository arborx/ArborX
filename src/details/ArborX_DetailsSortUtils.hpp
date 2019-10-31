/****************************************************************************
 * Copyright (c) 2012-2019 by the ArborX authors                            *
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

#include <ArborX_DetailsUtils.hpp> // iota
#include <ArborX_Exception.hpp>
#include <ArborX_Macros.hpp>

#include <Kokkos_Sort.hpp> // min_max_functor
#include <Kokkos_View.hpp>

#if defined(KOKKOS_ENABLE_CUDA)
#if (KOKKOS_COMPILER_CLANG < 900)
// Clang of version less than 9.0 cannot compile Thrust, failing with errors
// like this:
//    <snip>/thrust/system/cuda/detail/core/agent_launcher.h:557:11:
//    error: use of undeclared identifier 'va_printf'
// Defining _CubLog here allows us to avoid that code path, however disabling
// some debugging diagnostics
//
// If _CubLog is already defined, we save it into ARBORX_CubLog_save, and
// restore it at the end
#ifdef _CubLog
#define ARBORX_CubLog_save _CubLog
#endif
#define _CubLog(format, ...)
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#undef _CubLog
#ifdef ARBORX_CubLog_save
#define _CubLog ARBORX_CubLog_save
#undef ARBORX_CubLog_save
#endif
#else // #if (KOKKOS_COMPILER_CLANG < 900)
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#endif // #if (KOKKOS_COMPILER_CLANG < 900)
#endif // #if defined(KOKKOS_ENABLE_CUDA)

namespace ArborX
{

namespace Details
{

// NOTE returns the permutation indices **and** sorts the morton codes
template <typename DeviceType>
Kokkos::View<size_t *, DeviceType>
sortObjects(Kokkos::View<unsigned int *, DeviceType> view)
{
  using ExecutionSpace = typename DeviceType::execution_space;

  int const n = view.extent(0);

  using ViewType = decltype(view);
  using ValueType = typename ViewType::value_type;
  using CompType = Kokkos::BinOp1D<ViewType>;

  Kokkos::MinMaxScalar<ValueType> result;
  Kokkos::MinMax<ValueType> reducer(result);
  parallel_reduce(ARBORX_MARK_REGION("find_min_max_view"),
                  Kokkos::RangePolicy<ExecutionSpace>(0, n),
                  Kokkos::Impl::min_max_functor<ViewType>(view), reducer);
  if (result.min_val == result.max_val)
  {
    Kokkos::View<size_t *, DeviceType> permute(
        Kokkos::ViewAllocateWithoutInitializing("permute"), n);
    iota(permute);
    return permute;
  }

  // Passing the SizeType template argument to Kokkos::BinSort because it
  // defaults to the memory space size type which is different on the host and
  // on cuda (size_t versus unsigned int respectively).  size_t feels like a
  // better choice here because its size is guaranteed to coincide with the
  // pointer size which is a good thing for converting with reinterpret_cast
  // (when leaf indices are encoded into the pointer to one of their children)
  Kokkos::BinSort<ViewType, CompType, DeviceType, size_t> bin_sort(
      view, CompType(n / 2, result.min_val, result.max_val), true);
  bin_sort.create_permute_vector();
  bin_sort.sort(view);

  return bin_sort.get_permute_vector();
}

#if defined(KOKKOS_ENABLE_CUDA)
// NOTE returns the permutation indices **and** sorts the morton codes
template <typename MemorySpace>
Kokkos::View<size_t *, Kokkos::Device<Kokkos::Cuda, MemorySpace>> sortObjects(
    Kokkos::View<unsigned int *, Kokkos::Device<Kokkos::Cuda, MemorySpace>>
        view)
{
  int const n = view.extent(0);

  Kokkos::View<size_t *, Kokkos::Device<Kokkos::Cuda, MemorySpace>> permute(
      Kokkos::ViewAllocateWithoutInitializing("permutation"), n);
  ArborX::iota(permute);

  auto permute_ptr = thrust::device_ptr<size_t>(permute.data());
  auto begin_ptr = thrust::device_ptr<unsigned int>(view.data());
  auto end_ptr = thrust::device_ptr<unsigned int>(view.data() + n);
  thrust::sort_by_key(begin_ptr, end_ptr, permute_ptr);

  return permute;
}
#endif
} // namespace Details

} // namespace ArborX

#endif
