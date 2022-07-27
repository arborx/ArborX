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
#ifndef ARBORX_GEOMETRY_TRAITS_HPP
#define ARBORX_GEOMETRY_TRAITS_HPP

#include <Kokkos_Core.hpp>

namespace ArborX
{

namespace GeometryTraits
{

struct PointTag
{};

struct BoxTag
{};

struct SphereTag
{};

template <typename Geometry>
struct dimension
{
  using not_specialized = void; // tag to detect existence of a specialization
};

template <typename Geometry>
struct tag
{
  using not_specialized = void; // tag to detect existence of a specialization
};

template <typename Geometry>
using TagArchetypeAlias = typename tag<Geometry>::type;

template <typename Geometry>
struct is_point
    : Kokkos::is_detected_exact<PointTag, TagArchetypeAlias, Geometry>
{};

template <typename Geometry>
struct is_box : Kokkos::is_detected_exact<BoxTag, TagArchetypeAlias, Geometry>
{};

template <typename Geometry>
struct is_sphere
    : Kokkos::is_detected_exact<SphereTag, TagArchetypeAlias, Geometry>
{};

} // namespace GeometryTraits

} // namespace ArborX

#endif
