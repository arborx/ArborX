/****************************************************************************
 * Copyright (c) 2012-2019 by the ArborX authors                            *
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

template <typename... Ts>
struct ArrayTraits;

void checkProperlySized(std::size_t, std::size_t) {}

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

template <typename Array, typename... Arrays>
std::size_t getSizeOfArrays(Array const &v, Arrays const &... o)
{
  auto const s = ArrayTraits<Array>::size(v);
  checkProperlySized(1, s, o...);
  return s;
}

} // namespace Details

template <typename... Arrays>
std::vector<std::tuple<typename Details::ArrayTraits<Arrays>::value_type...>>
subsetToVectorOfTuples(std::size_t first, std::size_t last,
                       Arrays const &... in)
{
  std::vector<std::tuple<typename Details::ArrayTraits<Arrays>::value_type...>>
      out;
  for (std::size_t i = first; i < last; ++i)
  {
    out.emplace_back(Details::ArrayTraits<Arrays>::access(in, i)...);
  }
  return out;
}

#endif
