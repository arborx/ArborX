#ifndef ARBORX_TRIANGLE_HPP
#define ARBORX_TRIANGLE_HPP

#include <ArborX_Point.hpp>

namespace ArborX
{
namespace Experimental
{
// need to add a protection that
// the points are not on the same line.
struct Triangle
{
  Point a;
  Point b;
  Point c;
};
} // namespace Experimental
} // namespace ArborX

#endif
