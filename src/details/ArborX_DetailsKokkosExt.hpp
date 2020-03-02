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

#include <Kokkos_Concepts.hpp>
#include <Kokkos_View.hpp>

#include <cfloat>  // DBL_MAX, DBL_EPSILON
#include <cmath>   // isfinite, HUGE_VAL
#include <cstdint> // uint32_t
#include <type_traits>

#if __cplusplus < 201402L
namespace std
{
template <bool B, class T = void>
using enable_if_t = typename std::enable_if<B, T>::type;
} // namespace std
#endif

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
struct is_accessible_from<
    MemorySpace, ExecutionSpace,
    typename std::enable_if<Kokkos::Impl::SpaceAccessibility<
        ExecutionSpace, MemorySpace>::accessible>::type> : std::true_type
{
};

template <typename View>
struct is_accessible_from_host
    : public is_accessible_from<typename View::memory_space, Kokkos::HostSpace>
{
  static_assert(Kokkos::is_view<View>::value, "");
};

/** Count the number of consecutive leading zero bits in 32 bit integer
 * @param x
 */
KOKKOS_INLINE_FUNCTION
int clz(uint32_t x)
{
#if defined(__CUDA_ARCH__)
  // Note that the __clz() CUDA intrinsic function takes a signed integer
  // as input parameter.  This is fine but would need to be adjusted if
  // we were to change expandBits() and morton3D() to subdivide [0, 1]^3
  // into more 1024^3 bins.
  return __clz(x);
#elif defined(KOKKOS_COMPILER_GNU) || (KOKKOS_COMPILER_CLANG >= 500)
  // According to https://en.wikipedia.org/wiki/Find_first_set
  // Clang 5.X supports the builtin function with the same syntax as GCC
  return (x == 0) ? 32 : __builtin_clz(x);
#else
  if (x == 0)
    return 32;
  // The following is taken from:
  // http://stackoverflow.com/questions/23856596/counting-leading-zeros-in-a-32-bit-unsigned-integer-with-best-algorithm-in-c-pro
  const char debruijn32[32] = {0,  31, 9,  30, 3,  8,  13, 29, 2,  5, 7,
                               21, 12, 24, 28, 19, 1,  10, 4,  14, 6, 22,
                               25, 20, 11, 15, 23, 26, 16, 27, 17, 18};
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x++;
  return debruijn32[x * 0x076be629 >> 27];
#endif
}

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

/**
 * Branchless sign function. Return 1 if @param x is greater than zero, 0 if
 * @param x is zero, and -1 if @param x is less than zero.
 */
template <typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value>>
KOKKOS_INLINE_FUNCTION int sgn(T x)
{
  return (x > 0) - (x < 0);
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
