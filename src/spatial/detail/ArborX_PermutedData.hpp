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

template <typename Data, typename Permute, bool AttachIndices = false>
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
struct PermutedData<Data, Permute, /*AttachIndices=*/true>
{
  using memory_space = typename Data::memory_space;
  using value_type =
      std::decay_t<decltype(attach(std::declval<Data const &>()(0), 0))>;

  Data _data;
  Permute _permute;

  KOKKOS_FUNCTION decltype(auto) operator()(int i) const
  {
    return attach(_data(_permute(i)), i);
  }
  KOKKOS_FUNCTION auto size() const { return _data.size(); }
};

template <typename Data, typename Permute, bool AttachIndices>
class AccessValuesI<PermutedData<Data, Permute, AttachIndices>>
    : public PermutedData<Data, Permute, AttachIndices>
{
public:
  using self_type = PermutedData<Data, Permute, AttachIndices>;
};

} // namespace Details

template <Details::Concepts::Predicates Predicates, typename Permute,
          bool AttachIndices>
struct AccessTraits<Details::PermutedData<Predicates, Permute, AttachIndices>>
{
  using PermutedPredicates =
      Details::PermutedData<Predicates, Permute, AttachIndices>;

  using memory_space = typename Predicates::memory_space;

  KOKKOS_FUNCTION static std::size_t
  size(PermutedPredicates const &permuted_predicates)
  {
    return permuted_predicates.size();
  }

  KOKKOS_FUNCTION static decltype(auto)
  get(PermutedPredicates const &permuted_predicates, std::size_t index)
  {
    return permuted_predicates(index);
  }
};

} // namespace ArborX

#endif
