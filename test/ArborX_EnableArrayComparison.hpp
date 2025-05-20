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

#ifndef ARBORX_ENABLE_ARRAY_COMPARISON_HPP
#define ARBORX_ENABLE_ARRAY_COMPARISON_HPP

#include <Kokkos_Array.hpp>

#include "BoostTest_CUDA_clang_workarounds.hpp"
#include <boost/test/unit_test.hpp>
#include <boost/test/utils/is_forward_iterable.hpp>

namespace boost::unit_test
{

template <typename T, size_t N>
struct is_forward_iterable<Kokkos::Array<T, N>> : public boost::mpl::true_
{};

template <typename T, size_t N>
struct bt_iterator_traits<Kokkos::Array<T, N>, true>
{
  using array_type = Kokkos::Array<T, N>;
  using value_type = typename array_type::value_type;
  using const_iterator = typename array_type::const_pointer;
  static const_iterator begin(array_type const &v) { return v.data(); }
  static const_iterator end(array_type const &v) { return v.data() + v.size(); }
  static std::size_t size(array_type const &v) { return v.size(); }
};

} // namespace boost::unit_test

#endif
