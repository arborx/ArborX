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

namespace Experimental
{

struct PointTag
{};

struct BoxTag
{};

struct SphereTag
{};

template <typename Geometry, typename Enable = void>
struct GeometryTraits
{
  using not_specialized = void; // tag to detect existence of a specialization
};

template <typename Traits>
using GeometryTraitsTagArchetypeAlias = typename Traits::tag;

template <typename Geometry>
struct is_point
    : std::is_same<Kokkos::detected_t<GeometryTraitsTagArchetypeAlias,
                                      GeometryTraits<Geometry>>,
                   PointTag>::type
{};

template <typename Geometry>
struct is_box : std::is_same<Kokkos::detected_t<GeometryTraitsTagArchetypeAlias,
                                                GeometryTraits<Geometry>>,
                             BoxTag>::type
{};

template <typename Geometry>
struct is_sphere
    : std::is_same<Kokkos::detected_t<GeometryTraitsTagArchetypeAlias,
                                      GeometryTraits<Geometry>>,
                   SphereTag>::type
{};

} // namespace Experimental

} // namespace ArborX

#endif
