/****************************************************************************
 * Copyright (c) 2012-2018 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef DTK_BOOST_RANGE_ADAPTERS_HPP
#define DTK_BOOST_RANGE_ADAPTERS_HPP

#include <DTK_DBC.hpp>

#include <Kokkos_Concepts.hpp>
#include <Kokkos_View.hpp>

#include <type_traits> // is_same

namespace boost
{
// TODO SpaceAccessibility has been promoted to Kokkos:: namespace in later
// versions of Kokkos so eventually get rid of Impl::
#define DTK_ASSERT_VIEW_COMPATIBLE( View )                                     \
    using Traits = typename View::traits;                                      \
    static_assert(                                                             \
        Traits::rank == 1,                                                     \
        "Adaptor to Boost.Range only available for Views of rank 1" );         \
    static_assert(                                                             \
        Kokkos::Impl::SpaceAccessibility<                                      \
            Kokkos::HostSpace, typename Traits::memory_space>::accessible,     \
        "Adaptor to Boost.Range only available when View memory space is "     \
        "accessible from host" );

// Adapt Kokkos::View<T, P...> to Boost.Range
template <typename T, typename... P>
struct range_iterator<Kokkos::View<T, P...>>
{
    using View = Kokkos::View<T, P...>;
    DTK_ASSERT_VIEW_COMPATIBLE( View )
    typedef typename std::add_pointer<
        typename Kokkos::ViewTraits<T, P...>::value_type>::type type;
};

template <typename T, typename... P>
struct range_const_iterator<Kokkos::View<T, P...>>
{
    using View = Kokkos::View<T, P...>;
    DTK_ASSERT_VIEW_COMPATIBLE( View )
    typedef typename std::add_pointer<
        typename Kokkos::ViewTraits<T, P...>::const_value_type>::type type;
};

template <typename T, typename... P>
struct range_value<Kokkos::View<T, P...>>
{
    using View = Kokkos::View<T, P...>;
    DTK_ASSERT_VIEW_COMPATIBLE( View )
    typedef typename Kokkos::ViewTraits<T, P...>::value_type type;
};

template <typename T, typename... P>
struct range_reference<Kokkos::View<T, P...>>
{
    using View = Kokkos::View<T, P...>;
    DTK_ASSERT_VIEW_COMPATIBLE( View )
    typedef typename Kokkos::ViewTraits<T, P...>::reference_type type;
};

template <typename T, typename... P>
struct range_size<Kokkos::View<T, P...>>
{
    using View = Kokkos::View<T, P...>;
    DTK_ASSERT_VIEW_COMPATIBLE( View )
    typedef typename Kokkos::ViewTraits<T, P...>::size_type type;
};

} // end namespace boost

namespace Kokkos
{

template <typename T, typename... P>
inline typename boost::range_iterator<Kokkos::View<T, P...>>::type
range_begin( Kokkos::View<T, P...> &v )
{
    using View = Kokkos::View<T, P...>;
    DTK_ASSERT_VIEW_COMPATIBLE( View )
    return v.data();
}

template <typename T, typename... P>
inline typename boost::range_const_iterator<Kokkos::View<T, P...>>::type
range_begin( Kokkos::View<T, P...> const &v )
{
    using View = Kokkos::View<T, P...>;
    DTK_ASSERT_VIEW_COMPATIBLE( View )
    return v.data();
}

template <typename T, typename... P>
inline typename boost::range_iterator<Kokkos::View<T, P...>>::type
range_end( Kokkos::View<T, P...> &v )
{
    using View = Kokkos::View<T, P...>;
    DTK_ASSERT_VIEW_COMPATIBLE( View )
    DTK_REQUIRE( v.span_is_contiguous() );
    return v.data() + v.span();
}

template <typename T, typename... P>
inline typename boost::range_const_iterator<Kokkos::View<T, P...>>::type
range_end( Kokkos::View<T, P...> const &v )
{
    using View = Kokkos::View<T, P...>;
    DTK_ASSERT_VIEW_COMPATIBLE( View )
    DTK_REQUIRE( v.span_is_contiguous() );
    return v.data() + v.span();
}

// optional
template <typename T, typename... P>
inline typename boost::range_size<Kokkos::View<T, P...>>::type
range_calculate_size( Kokkos::View<T, P...> const &v )
{
    using View = Kokkos::View<T, P...>;
    DTK_ASSERT_VIEW_COMPATIBLE( View )
    return v.extent( 0 );
}

#undef DTK_ASSERT_VIEW_COMPATIBLE

} // end namespace Kokkos

#endif
