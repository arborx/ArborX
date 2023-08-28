#ifndef ARBORX_TRIANGLE_HPP
#define ARBORX_TRIANGLE_HPP

#include <ArborX_HyperPoint.hpp>

namespace ArborX::ExperimentalHyperGeometry
{
// need to add a protection that
// the points are not on the same line.
template <int DIM, class Coordinate = float>
struct Triangle
{
  ExperimentalHyperGeometry::Point<DIM, Coordinate> a;
  ExperimentalHyperGeometry::Point<DIM, Coordinate> b;
  ExperimentalHyperGeometry::Point<DIM, Coordinate> c;
};

template <int DIM, class Coordinate>
Triangle(ExperimentalHyperGeometry::Point<DIM, Coordinate>,
         ExperimentalHyperGeometry::Point<DIM, Coordinate>,
         ExperimentalHyperGeometry::Point<DIM, Coordinate>)
    -> Triangle<DIM, Coordinate>;

} // namespace ArborX::ExperimentalHyperGeometry

template <int DIM, class Coordinate>
struct ArborX::GeometryTraits::dimension<
    ArborX::ExperimentalHyperGeometry::Triangle<DIM, Coordinate>>
{
  static constexpr int value = DIM;
};
template <int DIM, class Coordinate>
struct ArborX::GeometryTraits::tag<
    ArborX::ExperimentalHyperGeometry::Triangle<DIM, Coordinate>>
{
  using type = TriangleTag;
};
template <int DIM, class Coordinate>
struct ArborX::GeometryTraits::coordinate_type<
    ArborX::ExperimentalHyperGeometry::Triangle<DIM, Coordinate>>
{
  using type = Coordinate;
};

#endif
