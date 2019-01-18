/****************************************************************************
 * Copyright (c) 2012-2019 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
#ifndef DTK_DETAILS_KOKKOS_EXT_HPP
#define DTK_DETAILS_KOKKOS_EXT_HPP

#include <Kokkos_View.hpp>

#include <type_traits>

#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace KokkosExt
{
template <typename View, typename = void>
struct is_accessible_from_host : std::false_type
{
    static_assert( Kokkos::is_view<View>::value, "" );
};

template <typename View>
struct is_accessible_from_host<
    View,
    typename std::enable_if<Kokkos::Impl::SpaceAccessibility<
        Kokkos::HostSpace, typename View::memory_space>::accessible>::type>
    : std::true_type
{
};
} // namespace KokkosExt
#endif // DOXYGEN_SHOULD_SKIP_THIS

#endif
