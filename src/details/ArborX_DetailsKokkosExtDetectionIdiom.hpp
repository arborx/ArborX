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

#ifndef ARBORX_DETAILS_KOKKOS_EXT_DETECTION_IDIOM_HPP
#define ARBORX_DETAILS_KOKKOS_EXT_DETECTION_IDIOM_HPP

#include <Kokkos_Macros.hpp>
#if KOKKOS_VERSION >= 30500
#include <Kokkos_DetectionIdiom.hpp>

namespace KokkosExt
{
using Kokkos::detected_t;
using Kokkos::is_detected;
} // namespace KokkosExt

#else
#include <type_traits>

#if !defined(__cpp_lib_void_t)
namespace std
{
template <typename...>
using void_t = void;
}
#endif

namespace KokkosExt
{

struct not_a_type
{
  not_a_type() = delete;
  ~not_a_type() = delete;
  not_a_type(not_a_type const &) = delete;
  void operator=(not_a_type const &) = delete;
};

// primary template handles all types not supporting the archetypal Op
template <class, template <class...> class Op, class... Args>
struct is_detected_impl : std::false_type
{
  using type = not_a_type;
};

// specialization recognizes and handles only types supporting Op
template <template <class...> class Op, class... Args>
struct is_detected_impl<std::void_t<Op<Args...>>, Op, Args...> : std::true_type
{
  using type = Op<Args...>;
};

template <template <class...> class Op, class... Args>
struct is_detected : is_detected_impl<void, Op, Args...>
{
};

template <template <class...> class Op, class... Args>
using detected_t = typename is_detected<Op, Args...>::type;

} // namespace KokkosExt
#endif

#endif
