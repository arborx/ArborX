/****************************************************************************
 * Copyright (c) 2023 by the ArborX authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#pragma once

#include <ArborX.hpp>

#include <type_traits>

namespace Details
{
template <typename T>
using inner_value_t = std::decay_t<std::invoke_result_t<
    decltype(ArborX::AccessTraits<T, ArborX::PrimitivesTag>::get), T const &,
    int>>;
} // namespace Details