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

#include <ArborX_DetailsConcepts.hpp>

#include <tuple>
#include <type_traits>

void test_concepts_compile_only()
{
  using ArborX::Details::first_template_parameter_t;
  static_assert(std::is_same<first_template_parameter_t<std::tuple<int, float>>,
                             int>::value,
                "");
}
