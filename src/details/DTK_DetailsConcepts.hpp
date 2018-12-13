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
#ifndef DTK_DETAILS_CONCEPTS_HPP
#define DTK_DETAILS_CONCEPTS_HPP

#include <DTK_DetailsAlgorithms.hpp>

#include <type_traits>

#if !defined( __cpp_lib_void_t )
namespace std
{
template <typename...>
using void_t = void;
}
#endif

namespace DataTransferKit
{
namespace Details
{

// Checks for existence of a free function that expands an object of type
// Geometry using an object of type Other
template <typename Geometry, typename Other, typename = void>
struct is_expandable : std::false_type
{
};

template <typename Geometry, typename Other>
struct is_expandable<
    Geometry, Other,
    std::void_t<decltype(
        expand( std::declval<Geometry &>(), std::declval<Other const &>() ) )>>
    : std::true_type
{
};

} // namespace Details
} // namespace DataTransferKit

#endif
