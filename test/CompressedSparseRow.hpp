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

#ifndef ARBORX_COMPRESSED_SPARSE_ROW_HPP
#define ARBORX_COMPRESSED_SPARSE_ROW_HPP

#include "VectorOfTuples.hpp"

namespace ext
{

// Extension for compile-time sequence of integers that defines a helper class
// template make_index_sequence_with_offset to simplify the creation of
// std::index_sequence type with Offset, Offset+1, Offset+2, Offset+Count-1 as
// Ints.
template <std::size_t Offset, std::size_t... Ints>
constexpr std::index_sequence<(Offset + Ints)...>
add_offset(std::index_sequence<Ints...>)
{
  return {};
}

template <std::size_t Offset, std::size_t Count>
struct make_index_sequence_with_offset
    : public decltype(add_offset<Offset>(std::make_index_sequence<Count>{}))
{
};

} // namespace ext

// Returns the number of elements in the arrays at the positions given by Is....
// The size is assumed to be the same for all these arrays. The return type here
// is std::size_t.
template <typename Tuple, std::size_t... Is>
auto getSizeOfArraysInTuple(Tuple const &t, std::index_sequence<Is...>)
{
  return Details::getSizeOfArrays(std::get<Is>(t)...);
}

// Checks that the last entry of the first array stored in the tuple t is equal
// to the number of entries in the arrays in the remaining tuple entries (they
// are all to be the same).
template <typename TupleOfArrays>
auto getNNZ(TupleOfArrays const &t)
{
  using ArrayType = typename std::tuple_element<0, TupleOfArrays>::type;
  using Traits = Details::ArrayTraits<typename std::remove_cv<ArrayType>::type>;
  constexpr std::size_t N = std::tuple_size<TupleOfArrays>::value;
  auto const nnz = getSizeOfArraysInTuple(
      t, ext::make_index_sequence_with_offset<1, N - 1>{});
  auto const last_element_of_first_array = static_cast<std::size_t>(
      Traits::access(std::get<0>(t), Traits::size(std::get<0>(t)) - 1));
  if (nnz != last_element_of_first_array)
  {
    std::stringstream msg;
    msg << "mismatch between number of entries in trailing arrays and last "
           "element of first array"
        << " (" << nnz << " != " << last_element_of_first_array << ')';
    throw std::invalid_argument(msg.str());
  }
  return nnz;
}

// Returns the size of the 1st array minus one.
template <typename TupleOfArrays>
auto getNumberOfRows(TupleOfArrays const &t)
{
  using ArrayType = typename std::tuple_element<0, TupleOfArrays>::type;
  using Traits = Details::ArrayTraits<typename std::remove_cv<ArrayType>::type>;
  return static_cast<std::size_t>(Traits::size(std::get<0>(t)) - 1);
}

// Repacks the entries [first, last) in the tuple entries provided by Is... into
// a std::vector of std::tuples and sorts this vector.
template <typename TupleOfArrays, std::size_t... Is>
auto subsetAndSort(TupleOfArrays const &t, std::size_t first, std::size_t last,
                   std::index_sequence<Is...>)
{
  auto out = subsetToVectorOfTuples(first, last, std::get<Is>(t)...);
  std::sort(std::begin(out), std::end(out));
  return out;
}

// Given a std::tuple of arrays return a vector of tuples for the entries in row
// i and sorts them. The function assumes that the starting indices for each row
// are given by the first array and that the actual data is stored in the
// remaining arrays.
template <typename TupleOfArrays>
auto extractRow(TupleOfArrays const &t, int i)
{
  using ArrayType = typename std::tuple_element<0, TupleOfArrays>::type;
  using Traits = Details::ArrayTraits<typename std::remove_cv<ArrayType>::type>;
  constexpr std::size_t N = std::tuple_size<TupleOfArrays>::value;
  // NOTE not checking that i does not exceed the number of rows (minus one)
  return subsetAndSort(t, Traits::access(std::get<0>(t), i),
                       Traits::access(std::get<0>(t), i + 1),
                       ext::make_index_sequence_with_offset<1, N - 1>{});
}

#endif
