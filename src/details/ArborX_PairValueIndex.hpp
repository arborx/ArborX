/****************************************************************************
 * Copyright (c) 2023 by the ArborX authors                                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_PAIR_VALUE_INDEX_HPP
#define ARBORX_PAIR_VALUE_INDEX_HPP

#include <ArborX_AccessTraits.hpp>

#include <Kokkos_Macros.hpp>

namespace ArborX
{

template <typename Value, typename Index = unsigned>
struct PairValueIndex
{
  static_assert(std::is_integral_v<Index>);

  using value_type = Value;
  using index_type = Index;

  Value value;
  Index index;
};

namespace Experimental
{
template <typename Values, typename Index>
class AttachIndices
{
private:
  using Data = Details::AccessValues<Values>;

public:
  Data _data;

  using memory_space = typename Data::memory_space;
  using value_type = PairValueIndex<typename Data::value_type, Index>;

  AttachIndices(Values const &values)
      : _data{values}
  {}

  KOKKOS_FUNCTION
  auto operator()(int i) const { return value_type{_data(i), Index(i)}; }

  KOKKOS_FUNCTION
  auto size() const { return _data.size(); }
};

template <typename Values, typename Index = unsigned>
auto attach_indices(Values const &values)
{
  return AttachIndices<Values, Index>{values};
}
} // namespace Experimental

} // namespace ArborX

template <typename Values, typename Index>
struct ArborX::AccessTraits<ArborX::Experimental::AttachIndices<Values, Index>,
                            ArborX::PrimitivesTag>
{
  using Self = ArborX::Experimental::AttachIndices<Values, Index>;

  using memory_space = typename Self::memory_space;

  KOKKOS_FUNCTION static auto size(Self const &values) { return values.size(); }
  KOKKOS_FUNCTION static decltype(auto) get(Self const &values, int i)
  {
    return values(i);
  }
};

#endif
