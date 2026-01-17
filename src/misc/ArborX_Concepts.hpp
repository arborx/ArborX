/****************************************************************************
 * Copyright (c) 2025, ArborX authors                                       *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
#ifndef ARBORX_CONCEPTS_HPP
#define ARBORX_CONCEPTS_HPP

#include <type_traits>

namespace ArborX::Details::Concepts
{

template <typename T>
struct is_complete_helper
{
  template <typename U>
  static auto test(U *) -> std::integral_constant<bool, sizeof(U) == sizeof(U)>;
  static auto test(...) -> std::false_type;
  using type = decltype(test((T *)0));
};

template <typename T>
struct is_complete : is_complete_helper<T>::type
{};

template <typename T>
constexpr bool is_complete_v = is_complete<T>::value;

} // namespace ArborX::Details::Concepts

#endif
