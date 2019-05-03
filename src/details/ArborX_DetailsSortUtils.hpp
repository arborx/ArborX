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

namespace ArborX
{

namespace Details
{

// NOTE returns the permutation indices **and** sorts the morton codes
template <typename DeviceType>
Kokkos::View<size_t *, DeviceType>
sortObjects(Kokkos::View<unsigned int *, DeviceType> view)
{
  int const n = view.extent(0);

  using ViewType = decltype(view);
  using ValueType = typename ViewType::value_type;
  using CompType = Kokkos::BinOp1D<ViewType>;
  using ExecutionSpace = typename DeviceType::execution_space;

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
  Kokkos::fence();

  return bin_sort.get_permute_vector();
}

} // namespace Details

} // namespace ArborX

#endif
