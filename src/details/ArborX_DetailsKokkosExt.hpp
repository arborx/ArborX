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
#ifndef ARBORX_DETAILS_KOKKOS_EXT_HPP
#define ARBORX_DETAILS_KOKKOS_EXT_HPP

#include <Kokkos_Core.hpp>

#include <cfloat>  // DBL_MAX, DBL_EPSILON
#include <cmath>   // isfinite, HUGE_VAL
#include <cstdint> // uint32_t
#include <type_traits>

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace KokkosExt
{
template <typename MemorySpace, typename ExecutionSpace, typename = void>
struct is_accessible_from : std::false_type
{
  static_assert(Kokkos::is_memory_space<MemorySpace>::value, "");
  static_assert(Kokkos::is_execution_space<ExecutionSpace>::value, "");
};

template <typename MemorySpace, typename ExecutionSpace>
struct is_accessible_from<MemorySpace, ExecutionSpace,
                          typename std::enable_if<Kokkos::SpaceAccessibility<
                              ExecutionSpace, MemorySpace>::accessible>::type>
    : std::true_type
{
};

template <typename View>
struct is_accessible_from_host
    : public is_accessible_from<typename View::memory_space, Kokkos::HostSpace>
{
  static_assert(Kokkos::is_view<View>::value, "");
};

//! Compute the maximum of two values.
template <typename T>
KOKKOS_INLINE_FUNCTION constexpr T const &max(T const &a, T const &b)
{
  return (a > b) ? a : b;
}

//! Compute the minimum of two values.
template <typename T>
KOKKOS_INLINE_FUNCTION constexpr T const &min(T const &a, T const &b)
{
  return (a < b) ? a : b;
}

/** Determine whether the given floating point argument @param x has finite
 * value.
 *
 * NOTE: Clang issues a warning if the std:: namespace is missing and nvcc
 * complains about calling a __host__ function from a __host__ __device__
 * function when it is present.
 */
template <typename T>
KOKKOS_INLINE_FUNCTION bool isFinite(T x)
{
#ifdef __CUDA_ARCH__
  return isfinite(x);
#else
  return std::isfinite(x);
#endif
}

namespace ArithmeticTraits
{

template <typename T>
struct infinity;

template <>
struct infinity<float>
{
  static constexpr float value = HUGE_VALF;
};

template <>
struct infinity<double>
{
  static constexpr double value = HUGE_VAL;
};

template <typename T>
struct max;

template <>
struct max<float>
{
  static constexpr float value = FLT_MAX;
};

template <>
struct max<double>
{
  static constexpr double value = DBL_MAX;
};

template <typename T>
struct epsilon;

template <>
struct epsilon<float>
{
  static constexpr float value = FLT_EPSILON;
};

template <>
struct epsilon<double>
{
  static constexpr double value = DBL_EPSILON;
};

} // namespace ArithmeticTraits

} // namespace KokkosExt
#endif // DOXYGEN_SHOULD_SKIP_THIS

#endif
