/****************************************************************************
 * Copyright (c) 2012-2020 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_VECTOR_OF_TUPLES_HPP
#define ARBORX_VECTOR_OF_TUPLES_HPP

namespace Details
{

// Type traits class that provides uniform interface to arrays.
// The definition below is just a prototype.  The template can be specialized
// for array-like types.
template <typename Array>
struct ArrayTraits
{
  // The type of the underlying array.
  using array_type = Array;
  // The type of the elements that can be accessed.
  using value_type = typename array_type::value_type;
  // Returns the number of elements in the array.
  static std::size_t size(array_type const &);
  // Access the element at specified location (read-only).
  static value_type const &access(array_type const &, std::size_t);
};

inline void checkProperlySized(std::size_t, std::size_t) {}

// Checks that all the arrays have size s. The only sensible value for pos is 1
// to get the appropriate error message when not calling it recursively.
template <typename Array, typename... Arrays>
void checkProperlySized(std::size_t pos, std::size_t s, Array const &v,
                        Arrays const &... o)
{
  if (s != v.size())
  {
    std::stringstream msg;
    // NOTE increment arg position to get 1-based indices
    msg << "vector size mismatch (argument " << ++pos << " has size "
        << v.size() << " != " << s << ")";
    throw std::invalid_argument(msg.str());
  }
  checkProperlySized(++pos, s, o...);
}

// This function assumes (and checks) that all arrays have the same size and
// returns that.
template <typename Array, typename... Arrays>
std::size_t getSizeOfArrays(Array const &v, Arrays const &... o)
{
  auto const s = ArrayTraits<Array>::size(v);
  checkProperlySized(1, s, o...);
  return s;
}

} // namespace Details

// Repacks the arrays in... into a std::vector of std::tuple. Each vector entry
// contains all the arrays' entries [first, last) at the same position (in
// order). This, of course, only makes sense if all the arrays have the same
// size.
template <typename... Arrays>
auto subsetToVectorOfTuples(std::size_t first, std::size_t last,
                            Arrays const &... in)
{
  if (first > last || last > Details::getSizeOfArrays(in...))
  {
    throw std::invalid_argument("precondition violated");
  }
  std::vector<std::tuple<typename Details::ArrayTraits<Arrays>::value_type...>>
      out;
  for (std::size_t i = first; i < last; ++i)
  {
    out.emplace_back(Details::ArrayTraits<Arrays>::access(in, i)...);
  }
  return out;
}

#endif
