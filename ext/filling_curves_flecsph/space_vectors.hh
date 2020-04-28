/*~--------------------------------------------------------------------------~*
 * Copyright (c) 2019 Triad National Security, LLC
 * All rights reserved.
 *~--------------------------------------------------------------------------~*/

/*~--------------------------------------------------------------------------~*
 *
 * /@@@@@@@@  @@           @@@@@@   @@@@@@@@ @@@@@@@  @@      @@
 * /@@/////  /@@          @@////@@ @@////// /@@////@@/@@     /@@
 * /@@       /@@  @@@@@  @@    // /@@       /@@   /@@/@@     /@@
 * /@@@@@@@  /@@ @@///@@/@@       /@@@@@@@@@/@@@@@@@ /@@@@@@@@@@
 * /@@////   /@@/@@@@@@@/@@       ////////@@/@@////  /@@//////@@
 * /@@       /@@/@@//// //@@    @@       /@@/@@      /@@     /@@
 * /@@       @@@//@@@@@@ //@@@@@@  @@@@@@@@ /@@      /@@     /@@
 * //       ///  //////   //////  ////////  //       //      //
 *
 *~--------------------------------------------------------------------------~*/

/**
 * @file tensor.h
 * @author Oleg Korobkin
 * @date November 2019
 * @brief General tensors class
 */

#ifndef SPACE_VECTOR_H
#define SPACE_VECTOR_H

#include "tensors.hh"

namespace flecsi {

template<typename T, size_t D>
using space_vector_u = tensor_u<T, symmetry_type::generic, D>;

//----------------------------------------------------------------------------//
//! Returns the distance from point A to point B.
//!
//! @tparam T  The type to use to represent coordinate values.
//! @tparam D  The dimension of the point.
//!
//! @param a   Point A.
//! @param b   Point B.
//!
//----------------------------------------------------------------------------//
template<typename T, size_t D>
T
distance(space_vector_u<T, D> const & a, space_vector_u<T, D> const & b) {
  if constexpr(D == 1)
    return std::abs(a[0] - b[0]);

  T sum(0);
  for(size_t d(0); d < D; ++d) {
    sum += (a[d] - b[d])*(a[d] - b[d]);
  } // for

  return std::sqrt(sum);
} // distance

//----------------------------------------------------------------------------//
//! Return the midpoint between two points.
//!
//! @tparam TYPE      The type to use to represent coordinate values.
//! @tparam DIMENSION The dimension of the point.
//!
//! @param a The first point.
//! @param b The second point.
//!
//! @ingroup geometry
//----------------------------------------------------------------------------//
template<typename T, size_t D>
space_vector_u<T, D>
midpoint(space_vector_u<T, D> const & a, space_vector_u<T, D> const & b) {
  return space_vector_u<T, D>(0.5 * (a + b));
} // midpoint

//----------------------------------------------------------------------------//
//! Return the centroid of the given set of points.
//!
//! @tparam CONTAINER  Container class
//! @tparam T          The type to use to represent coordinate values.
//! @tparam D          The dimension of the point.
//!
//! @param points The points for which to find the centroid.
//!
//! @ingroup geometry
//----------------------------------------------------------------------------//

template<template<typename...> class CONTAINER, typename T, size_t D>
auto
centroid(CONTAINER<space_vector_u<T, D>> const & points) {
  space_vector_u<T, D> tmp(0.0);

  for(auto p : points) {
    tmp += p;
  } // for

  tmp /= points.size();

  return tmp;
} // centroid

/*!
  \function dot
 */
template<typename T, size_t D>
T
dot(const space_vector_u<T, D> & a, const space_vector_u<T, D> & b) {
  T sum(0);

  for(size_t d(0); d < D; ++d) {
    sum += a[d] * b[d];
  } // for

  return sum;
} // dot

/*!
  \function magnitude
 */
template<typename T, size_t D>
T
magnitude(const space_vector_u<T, D> & a) {
  if constexpr(D == 1)
    return std::abs(a[0]);

  T sum(0);
  for(size_t d(0); d < D; ++d) {
    sum += a[d]*a[d];
  } // for
  return std::sqrt(sum);
} // magnitude

/*
 \function cross product (D = 1)
 */
template<typename T>
space_vector_u<T, 1>
cross(const space_vector_u<T, 1> & a, const space_vector_u<T, 1> & b) {
  space_vector_u<T, 1> c{0.0};
  return c;
}

/*
 \function cross product (D = 2)
 */
template<typename T>
space_vector_u<T, 2>
cross(const space_vector_u<T, 2> & a, const space_vector_u<T, 2> & b) {
  space_vector_u<T, 2> c{0.0};
  return c;
}

/*
 \function cross product (D = 3)
 */
template<typename T>
space_vector_u<T, 3>
cross(const space_vector_u<T, 3> & a, const space_vector_u<T, 3> & b) {
  T cx = a[1] * b[2] - a[2] * b[1];
  T cy = a[2] * b[0] - a[0] * b[2];
  T cz = a[0] * b[1] - a[1] * b[0];
  space_vector_u<T, 3> c{cx, cy, cz};
  return c;
}

/*!
  \function cross_magnitude
 */
template<typename T>
T
cross_magnitude(const space_vector_u<T, 1> & a,
  const space_vector_u<T, 1> & b) {
  return 0.0;
} // cross_magnitude

/*!
  \function cross_magnitude
 */
template<typename T>
T
cross_magnitude(const space_vector_u<T, 2> & a,
  const space_vector_u<T, 2> & b) {
  return fabs(a[0] * b[1] - a[1] * b[0]);
} // cross_magnitude

/*!
  \function cross_magnitude
 */
template<typename T>
T
cross_magnitude(const space_vector_u<T, 3> & a,
  const space_vector_u<T, 3> & b) {
  return magnitude(cross(a, b));
} // cross_magnitude

/*!
  \function normal
  \brief for a vector xi + yj the normal vector is -yi + xj. given points
  a and b we use x = b[0] - a[0] and y = b[1] - a[1]
 */
template<typename T>
space_vector_u<T, 2>
normal(const space_vector_u<T, 2> & a, const space_vector_u<T, 2> & b) {
  space_vector_u<T, 2> tmp;
  tmp[0] = a[1] - b[1];
  tmp[1] = b[0] - a[0];
  return tmp;
} // normal

/*!
  \function normal
 */
template<typename T>
space_vector_u<T, 3>
normal(const space_vector_u<T, 3> & a, const space_vector_u<T, 3> & b) {
  space_vector_u<T, 3> tmp;
  tmp[0] = a[1] * b[2] - a[2] * b[1];
  tmp[1] = a[2] * b[0] - a[0] * b[2];
  tmp[2] = a[0] * b[1] - a[1] * b[0];
  return tmp;
} // normal
} // namespace flecsi

#endif // SPACE_VECTOR_H

/*
// Usage example
#include <iostream>

int main() {

  using namespace std;
  using namespace flecsph;
  using namespace tensor_indices;

  cout << "--- Vectors: -------------------" << endl;
  using vector_1d_t = space_vector_u<double, 1>;
  vector_1d_t b1;
  cout << "1D: ";
  b1[0]  = 1.8; cout << "b1[0]  = " << b1[0]  << "\n";
  cout << "    ";
  b1[_x] = 1.9; cout << "b1[_x] = " << b1[_x] << "\n";
  cout << "    ";
  b1(0)  = 2.1; cout << "b1(0)  = " << b1(0)  << "\n";

  using vector_2d_t = space_vector_u<double, 2>;
  vector_2d_t b2;
  cout << "2D: ";
  b2[0]  = 1.8; b2[1]  = -1.8;
  cout << "{b2[0],  b2[1] } = {" << b2[0]  << ", " << b2[1] << "}\n";
  cout << "    ";
  b2[_x] = 1.9; b2[_y] = -1.9;
  cout << "{b2[_x], b2[_y]} = {" << b2[_x] << ", " << b2[_y] << "}\n";
  cout << "    ";
  b2(0)  = 2.1; b2(1)  = -2.1;
  cout << "{b2(0),  b2(1) } = {" << b2(0)  << ", " << b2(1) << "}\n";

  using vector_3d_t = space_vector_u<double, 3>;
  vector_3d_t b3;
  cout << "3D: ";
  b3[0]  = 1.1; b3[1]  = 1.2; b3[2] = 1.3;
  cout <<"{b3[0 .. 2]} = {"<<b3[0]<<", "<<b3[1]<<", "<<b3[2]<< "}\n";
  cout << "    ";
  b3[_x] = 2.1; b3[_y] = 2.2; b3[_z] = 2.3;
  cout << "{b3[_x.._z]} = {"<<b3[_x]<<", "<<b3[_y]<<", "<<b3[_z]<< "}\n";
  cout << "    ";
  b3(0)  = 3.1; b3(1)  = 3.2; b3(2)  = 3.3;
  cout <<"{b3(0 .. 2)} = {"<<b3(0)<<", "<<b3(1)<<", "<<b3(2)<< "}\n";


}
*/
