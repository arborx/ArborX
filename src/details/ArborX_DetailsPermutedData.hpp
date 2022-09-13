/****************************************************************************
 * Copyright (c) 2017-2022 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_DETAILS_PERMUTED_DATA_HPP
#define ARBORX_DETAILS_PERMUTED_DATA_HPP

#include <ArborX_AccessTraits.hpp>

namespace ArborX
{

namespace Details
{

template <typename Data, typename Permute, bool AttachIndices = false>
struct PermutedData
{
  Data _data;
  Permute _permute;

  using memory_space = typename Data::memory_space;
  using value_type = typename Data::value_type;

  KOKKOS_FUNCTION decltype(auto) operator()(int i) const
  {
    return _data(_permute(i));
  }
  KOKKOS_FUNCTION decltype(auto) size() const { return _data.size(); }
};

} // namespace Details

} // namespace ArborX

#endif
