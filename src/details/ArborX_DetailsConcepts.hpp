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

#ifndef ARBORX_DETAILS_CONCEPTS_HPP
#define ARBORX_DETAILS_CONCEPTS_HPP

#include <type_traits>

#if !defined(__cpp_lib_void_t)
namespace std
{
template <typename...>
using void_t = void;
}
#endif

#if !defined(DOXYGEN_SHOULD_SKIP_THIS)
namespace ArborX
{
namespace Details
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

template <typename T>
struct first_template_parameter;

template <template <typename...> class E, typename Head, typename... Tail>
struct first_template_parameter<E<Head, Tail...>>
{
  using type = Head;
};

template <typename T>
using first_template_parameter_t = typename first_template_parameter<T>::type;

} // namespace Details
} // namespace ArborX
#endif // DOXYGEN_SHOULD_SKIP_THIS

#endif
