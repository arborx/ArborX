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

void checkSize(std::size_t, std::size_t) {}

template <typename U, typename... Ts>
void checkSize(std::size_t pos, std::size_t s, std::vector<U> const &v,
               std::vector<Ts> const &... o)
{
  if (s != v.size())
  {
    std::stringstream msg;
    // NOTE increment arg position to get 1-based indices
    msg << "vector size mismatch (argument " << ++pos << " has size "
        << v.size() << " != " << s << ")";
    throw std::invalid_argument(msg.str());
  }
  checkSize(++pos, s, o...);
}

template <typename U, typename... Ts>
std::size_t getSize(std::vector<U> const &v, std::vector<Ts> const &... o)
{
  auto const s = v.size();
  checkSize(1, s, o...);
  return s;
}

} // namespace Details

template <typename... Ts>
std::vector<std::tuple<Ts...>> toVectorOfTuples(std::vector<Ts> const &... in)
{
  std::vector<std::tuple<Ts...>> out;
  std::size_t const n = Details::getSize(in...);
  for (std::size_t i = 0; i < n; ++i)
  {
    out.emplace_back(in[i]...);
  }
  return out;
}

#endif
