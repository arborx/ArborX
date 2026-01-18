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

#ifndef ARBORX_DETAILS_GEOMETRY_REVERSE_DISPATCH_HPP
#define ARBORX_DETAILS_GEOMETRY_REVERSE_DISPATCH_HPP

#include <Kokkos_Macros.hpp>

#include <concepts>

namespace ArborX::Details::Dispatch
{

template <typename Algorithm, typename Geometry1, typename Geometry2>
concept CanApply =
    requires(Geometry1 const &geometry1, Geometry2 const &geometry2) {
      {
        Algorithm::apply(geometry1, geometry2)
      };
    };

template <template <typename, typename, typename, typename> typename Algorithm,
          typename Tag1, typename Tag2, typename Geometry1, typename Geometry2>
struct DoApply
{
  using ForwardAlgorithm = Algorithm<Tag1, Tag2, Geometry1, Geometry2>;
  using ReverseAlgorithm = Algorithm<Tag2, Tag1, Geometry2, Geometry1>;
  static KOKKOS_FUNCTION constexpr auto apply(Geometry1 const &geometry1,
                                              Geometry2 const &geometry2)
  {
    if constexpr (!std::same_as<Geometry1, Geometry2> &&
                  !CanApply<ForwardAlgorithm, Geometry1, Geometry2> &&
                  CanApply<ReverseAlgorithm, Geometry2, Geometry1>)
      return ReverseAlgorithm::apply(geometry2, geometry1);
    else
      return ForwardAlgorithm::apply(geometry1, geometry2);
  }
};

} // namespace ArborX::Details::Dispatch

#endif
