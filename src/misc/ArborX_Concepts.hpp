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

// The following would be a nicer version not requiring lambdas:
//   template <typename T, bool value = requires(T) { sizeof(T); }>
//   constexpr bool is_complete_v = value;
// However, it does not compile on Cuda: "mangling for "requires"
// expressions is not yet implemented" (Cuda 12.9)

template <class T, auto _x = [] {}>
concept is_complete_v = requires {
  sizeof(T);
  _x;
};
} // namespace ArborX::Details::Concepts

#endif
