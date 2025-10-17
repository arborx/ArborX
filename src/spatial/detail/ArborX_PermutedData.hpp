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

#ifndef ARBORX_PERMUTED_DATA_HPP
#define ARBORX_PERMUTED_DATA_HPP

#include <detail/ArborX_AccessTraits.hpp>

namespace ArborX
{

namespace Details
{

template <typename Data, typename Permute>
struct PermutedData
{
  using memory_space = typename Data::memory_space;
  using value_type = typename Data::value_type;

  Data _data;
  Permute _permute;

  KOKKOS_FUNCTION decltype(auto) operator()(int i) const
  {
    return _data(_permute(i));
  }
  KOKKOS_FUNCTION auto size() const { return _data.size(); }
};

template <typename Data, typename Permute>
class AccessValuesI<PermutedData<Data, Permute>>
    : public PermutedData<Data, Permute>
{
public:
  using self_type = PermutedData<Data, Permute>;
};

} // namespace Details

template <typename Data, typename Permute>
struct AccessTraits<Details::PermutedData<Data, Permute>>
{
  using Self = Details::PermutedData<Data, Permute>;

  using memory_space = typename Data::memory_space;

  KOKKOS_FUNCTION static std::size_t size(Self const &permuted_data)
  {
    return permuted_data.size();
  }

  KOKKOS_FUNCTION static decltype(auto) get(Self const &permuted_data,
                                            std::size_t index)
  {
    return permuted_data(index);
  }
};

} // namespace ArborX

#endif
