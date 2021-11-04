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

#ifndef ARBORX_DETAILS_KOKKOS_EXT_ARITHMETIC_TRAITS_HPP
#define ARBORX_DETAILS_KOKKOS_EXT_ARITHMETIC_TRAITS_HPP

#include <cfloat> // DBL_MAX, DBL_EPSILON
#include <cmath>  // HUGE_VAL
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

} // namespace Details

template <class T>
struct infinity : Details::infinity<std::remove_cv_t<T>>
{
};

template <class T>
struct max : Details::max<std::remove_cv_t<T>>
{
};

template <class T>
struct epsilon : Details::epsilon<std::remove_cv_t<T>>
{
};

} // namespace ArithmeticTraits

} // namespace KokkosExt

#endif
