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

#ifndef ARBORX_BOOST_RANGE_ADAPTERS_HPP
#define ARBORX_BOOST_RANGE_ADAPTERS_HPP

#include <ArborX_DetailsKokkosExtAccessibilityTraits.hpp>
#include <ArborX_Exception.hpp>

#include <Kokkos_Core.hpp>

#include <boost/range.hpp>

#include <type_traits> // is_same

namespace boost
{
#define ARBORX_ASSERT_VIEW_COMPATIBLE(View)                                    \
  using Traits = typename View::traits;                                        \
  static_assert(Traits::rank == 1,                                             \
                "Adaptor to Boost.Range only available for Views of rank 1");  \
  static_assert(                                                               \
      KokkosExt::is_accessible_from_host<View>::value,                         \
      "Adaptor to Boost.Range only available when View memory space is "       \
      "accessible from host");

// Adapt Kokkos::View<T, P...> to Boost.Range
template <typename T, typename... P>
struct range_iterator<Kokkos::View<T, P...>>
{
  using View = Kokkos::View<T, P...>;
  ARBORX_ASSERT_VIEW_COMPATIBLE(View)
  using type = typename std::add_pointer<
      typename Kokkos::ViewTraits<T, P...>::value_type>::type;
};

template <typename T, typename... P>
struct range_const_iterator<Kokkos::View<T, P...>>
{
  using View = Kokkos::View<T, P...>;
  ARBORX_ASSERT_VIEW_COMPATIBLE(View)
  using type = typename std::add_pointer<
      typename Kokkos::ViewTraits<T, P...>::const_value_type>::type;
};

template <typename T, typename... P>
struct range_value<Kokkos::View<T, P...>>
{
  using View = Kokkos::View<T, P...>;
  ARBORX_ASSERT_VIEW_COMPATIBLE(View)
  using type = typename Kokkos::ViewTraits<T, P...>::value_type;
};

template <typename T, typename... P>
struct range_reference<Kokkos::View<T, P...>>
{
  using View = Kokkos::View<T, P...>;
  ARBORX_ASSERT_VIEW_COMPATIBLE(View)
  using type = typename Kokkos::ViewTraits<T, P...>::reference_type;
};

template <typename T, typename... P>
struct range_size<Kokkos::View<T, P...>>
{
  using View = Kokkos::View<T, P...>;
  ARBORX_ASSERT_VIEW_COMPATIBLE(View)
  using type = typename Kokkos::ViewTraits<T, P...>::size_type;
};

} // namespace boost

namespace Kokkos
{

template <typename T, typename... P>
inline typename boost::range_iterator<Kokkos::View<T, P...>>::type
range_begin(Kokkos::View<T, P...> &v)
{
  using View = Kokkos::View<T, P...>;
  ARBORX_ASSERT_VIEW_COMPATIBLE(View)
  return v.data();
}

template <typename T, typename... P>
inline typename boost::range_const_iterator<Kokkos::View<T, P...>>::type
range_begin(Kokkos::View<T, P...> const &v)
{
  using View = Kokkos::View<T, P...>;
  ARBORX_ASSERT_VIEW_COMPATIBLE(View)
  return v.data();
}

template <typename T, typename... P>
inline typename boost::range_iterator<Kokkos::View<T, P...>>::type
range_end(Kokkos::View<T, P...> &v)
{
  using View = Kokkos::View<T, P...>;
  ARBORX_ASSERT_VIEW_COMPATIBLE(View)
  ARBORX_ASSERT(v.span_is_contiguous());
  return v.data() + v.span();
}

template <typename T, typename... P>
inline typename boost::range_const_iterator<Kokkos::View<T, P...>>::type
range_end(Kokkos::View<T, P...> const &v)
{
  using View = Kokkos::View<T, P...>;
  ARBORX_ASSERT_VIEW_COMPATIBLE(View)
  ARBORX_ASSERT(v.span_is_contiguous());
  return v.data() + v.span();
}

// optional
template <typename T, typename... P>
inline typename boost::range_size<Kokkos::View<T, P...>>::type
range_calculate_size(Kokkos::View<T, P...> const &v)
{
  using View = Kokkos::View<T, P...>;
  ARBORX_ASSERT_VIEW_COMPATIBLE(View)
  return v.extent(0);
}

#undef ARBORX_ASSERT_VIEW_COMPATIBLE

} // namespace Kokkos

#endif
