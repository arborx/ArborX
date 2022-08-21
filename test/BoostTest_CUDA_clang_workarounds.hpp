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

#ifndef BOOSTTEST_CUDA_CLANG_WORKAROUNDS_HPP
#define BOOSTTEST_CUDA_CLANG_WORKAROUNDS_HPP

#include <boost/fusion/adapted/std_tuple.hpp>
#include <boost/fusion/include/std_tuple.hpp>
#include <boost/type_traits/add_const.hpp>
#include <boost/type_traits/add_pointer.hpp>

#ifndef BOOST_STATIC_ASSERT
#define BOOST_STATIC_ASSERT(m) static_assert(m);
#endif

#endif
