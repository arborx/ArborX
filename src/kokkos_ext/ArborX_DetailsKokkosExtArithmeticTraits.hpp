/****************************************************************************
 * Copyright (c) 2017-2022 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_DETAILS_KOKKOS_EXT_ARITHMETIC_TRAITS_HPP
#define ARBORX_DETAILS_KOKKOS_EXT_ARITHMETIC_TRAITS_HPP

#include <Kokkos_Macros.hpp>
#if KOKKOS_VERSION >= 30599
#include <Kokkos_NumericTraits.hpp>
namespace KokkosExt
{
namespace ArithmeticTraits
{
template <class T>
using infinity = Kokkos::Experimental::infinity<T>;

template <class T>
using finite_max = Kokkos::Experimental::finite_max<T>;

template <class T>
using finite_min = Kokkos::Experimental::finite_min<T>;

template <class T>
using epsilon = Kokkos::Experimental::epsilon<T>;
} // namespace ArithmeticTraits
} // namespace KokkosExt

#else
#include <cfloat>  // DBL_MAX, DBL_EPSILON
#include <climits> // INT_MAX, INT_MIN
#include <cmath>   // HUGE_VAL
#include <type_traits>

namespace KokkosExt
{
namespace ArithmeticTraits
{
namespace Details
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
struct finite_max;

template <>
struct finite_max<float>
{
  static constexpr float value = FLT_MAX;
};

template <>
struct finite_max<double>
{
  static constexpr double value = DBL_MAX;
};

template <>
struct finite_max<int>
{
  static constexpr int value = INT_MAX;
};

template <>
struct finite_max<long long>
{
  static constexpr long long value = LLONG_MAX;
};

template <typename T>
struct finite_min;

template <>
struct finite_min<float>
{
  static constexpr float value = -FLT_MAX;
};

template <>
struct finite_min<double>
{
  static constexpr double value = -DBL_MAX;
};

template <>
struct finite_min<int>
{
  static constexpr int value = INT_MIN;
};

template <>
struct finite_min<long long>
{
  static constexpr long long value = LLONG_MIN;
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

} // namespace Details

template <class T>
struct infinity : Details::infinity<std::remove_cv_t<T>>
{};

template <class T>
struct finite_max : Details::finite_max<std::remove_cv_t<T>>
{};

template <class T>
struct finite_min : Details::finite_min<std::remove_cv_t<T>>
{};

template <class T>
struct epsilon : Details::epsilon<std::remove_cv_t<T>>
{};

} // namespace ArithmeticTraits

} // namespace KokkosExt
#endif

#endif
