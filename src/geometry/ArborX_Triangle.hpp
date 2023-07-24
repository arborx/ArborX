#ifndef ARBORX_TRIANGLE_HPP
#define ARBORX_TRIANGLE_HPP

#include <ArborX_HyperPoint.hpp>

namespace ArborX
{
namespace Experimental
{
// need to add a protection that
// the points are not on the same line.
template <int DIM, class FloatingPoint = float>
struct Triangle
{
  ExperimentalHyperGeometry::Point<DIM, FloatingPoint> a;
  ExperimentalHyperGeometry::Point<DIM, FloatingPoint> b;
  ExperimentalHyperGeometry::Point<DIM, FloatingPoint> c;
};

template <int DIM, class FloatingPoint>
Triangle(ExperimentalHyperGeometry::Point<DIM, FloatingPoint>,
         ExperimentalHyperGeometry::Point<DIM, FloatingPoint>,
         ExperimentalHyperGeometry::Point<DIM, FloatingPoint>)
    -> Triangle<DIM, FloatingPoint>;

} // namespace Experimental
} // namespace ArborX

template <int DIM, class FloatingPoint>
struct ArborX::GeometryTraits::dimension<
    ArborX::Experimental::Triangle<DIM, FloatingPoint>>
{
  static constexpr int value = DIM;
};
#endif
