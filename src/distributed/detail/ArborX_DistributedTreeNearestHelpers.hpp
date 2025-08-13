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
#ifndef ARBORX_DISTRIBUTED_TREE_NEAREST_HELPERS_HPP
#define ARBORX_DISTRIBUTED_TREE_NEAREST_HELPERS_HPP

#include <ArborX_Point.hpp>
#include <ArborX_Sphere.hpp>
#include <algorithms/ArborX_Convert.hpp>
#include <detail/ArborX_AccessTraits.hpp>
#include <detail/ArborX_Callbacks.hpp>

#include <Kokkos_BitManipulation.hpp>

namespace ArborX
{

namespace Experimental
{

// Constrained callback is a callback that a user promises to:
// - be not pure (i.e., allow output argument in the signature)
// - be allowed to be called on non-final results
// - produce exactly one result for each match
template <class Callback>
struct ConstrainedDistributedNearestCallback
{
  Callback _callback;

  template <class... Args>
  KOKKOS_FUNCTION void operator()(Args &&...args) const
  {
    _callback((Args &&)args...);
  }
};

template <class Callback>
auto declare_callback_constrained(Callback const &callback)
{
  return ConstrainedDistributedNearestCallback<Callback>{callback};
}

} // namespace Experimental

namespace Details
{

template <class Callback>
struct is_constrained_callback : std::false_type
{};
template <class Callback>
struct is_constrained_callback<
    Experimental::ConstrainedDistributedNearestCallback<Callback>>
    : std::true_type
{};
// We need DefaultCallback specialization for the case without a user-provided
// callback and using APIv2
template <>
struct is_constrained_callback<DefaultCallback> : std::true_type
{};

template <class Callback>
inline constexpr bool is_constrained_callback_v =
    is_constrained_callback<Callback>::value;

template <class Predicates, class Distances>
struct WithinDistanceFromPredicates
{
  Predicates predicates;
  Distances distances;
};

namespace Dispatch
{

template <typename Tag, typename Geometry>
struct approx_expand_by_radius;

template <typename Point>
struct approx_expand_by_radius<PointTag, Point>
{
  template <typename Float>
  KOKKOS_FUNCTION static auto apply(Point const &point, Float r)
  {
    constexpr int DIM = GeometryTraits::dimension_v<Point>;
    using Coordinate = GeometryTraits::coordinate_type_t<Point>;
    return Sphere{Details::convert<::ArborX::Point<DIM, Coordinate>>(point), r};
  }
};

template <typename Box>
struct approx_expand_by_radius<BoxTag, Box>
{
  template <typename Float>
  KOKKOS_FUNCTION static auto apply(Box const &box, Float r)
  {
    using namespace KokkosExt::ArithmeticTraits;

    Box new_box = box;
    auto &min_corner = new_box.minCorner();
    auto &max_corner = new_box.maxCorner();
    constexpr int DIM = GeometryTraits::dimension_v<Box>;
    using Coordinate = GeometryTraits::coordinate_type_t<Box>;
    for (int d = 0; d < DIM; ++d)
    {
      min_corner[d] =
          Kokkos::nextafter(min_corner[d] - r, finite_min<Coordinate>::value);
      max_corner[d] =
          Kokkos::nextafter(max_corner[d] + r, finite_max<Coordinate>::value);
    }
    return new_box;
  }
};

template <typename Sphere>
struct approx_expand_by_radius<SphereTag, Sphere>
{
  template <typename Float>
  KOKKOS_FUNCTION static auto apply(Sphere const &sphere, Float r)
  {
    using namespace KokkosExt::ArithmeticTraits;
    using Coordinate = GeometryTraits::coordinate_type_t<Sphere>;
    return Sphere{
        sphere.centroid(),
        Kokkos::nextafter(sphere.radius() + r, finite_max<Coordinate>::value)};
  }
};

template <typename Ray>
struct approx_expand_by_radius<RayTag, Ray>
{
  template <typename Float>
  KOKKOS_FUNCTION static auto const &apply(Ray const &ray, Float)
  {
    return ray;
  }
};

} // namespace Dispatch

template <typename Geometry, typename Float>
KOKKOS_INLINE_FUNCTION auto approx_expand_by_radius(Geometry const &geometry,
                                                    Float r)
{
  return Dispatch::approx_expand_by_radius<
      typename GeometryTraits::tag_t<Geometry>, Geometry>::apply(geometry, r);
}

} // namespace Details

template <class Predicates, class Distances>
struct AccessTraits<
    Details::WithinDistanceFromPredicates<Predicates, Distances>>
{
  using Predicate = typename Predicates::value_type;
  using Geometry =
      std::decay_t<decltype(getGeometry(std::declval<Predicate const &>()))>;
  using Self = Details::WithinDistanceFromPredicates<Predicates, Distances>;

  using memory_space = typename Predicates::memory_space;
  using size_type = decltype(std::declval<Predicates const &>().size());

  static KOKKOS_FUNCTION size_type size(Self const &x)
  {
    return x.predicates.size();
  }
  static KOKKOS_FUNCTION auto get(Self const &x, size_type i)
  {
    using Details::approx_expand_by_radius;
    return intersects(
        approx_expand_by_radius(getGeometry(x.predicates(i)), x.distances(i)));
  }
};

namespace Details
{

template <typename Tree, typename Callback, typename OutValue, bool UseValues>
struct CallbackWithDistance
{
  Tree _tree;
  Callback _callback;

  template <typename ExecutionSpace>
  CallbackWithDistance(ExecutionSpace const &, Tree const &tree,
                       Callback const &callback)
      : _tree(tree)
      , _callback(callback)
  {}

  template <typename Query, typename Value, typename Output>
  KOKKOS_FUNCTION void operator()(Query const &query, Value const &value,
                                  Output const &out) const
  {
    if constexpr (UseValues)
    {
      OutValue out_value;
      [[maybe_unused]] int count = 0;
      _callback(query, value, [&](OutValue const &ov) {
        out_value = ov;
        // NOTE: this will break if we are running multiple threads per query.
        // It could happen that the callback is called by different threads,
        // resulting in multiple outputs while having count = 1 in each thread.
        // As we don't envision this happening in the near future, it is ok for
        // now.
        ++count;
      });
      // If the user callback produces no output, we have nothing to attach the
      // distance to, which is problematic as we would not be able to do the
      // final filtering. If there are multiple outputs, it will currently
      // break our communication routines and filtering. We rely on 3-phase
      // nearest implementation for these cases.
      KOKKOS_ASSERT(count == 1);
      out({out_value,
           distance(getGeometry(query), _tree.indexable_get()(value))});
    }
    else
      out(distance(getGeometry(query), _tree.indexable_get()(value)));
  }
};

} // namespace Details
} // namespace ArborX

#endif
