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
#ifndef ARBORX_DETAILS_ATTACH_INDICES_HPP
#define ARBORX_DETAILS_ATTACH_INDICES_HPP

#include <detail/ArborX_AccessTraits.hpp>
#include <detail/ArborX_PairValueIndex.hpp>
#include <detail/ArborX_Predicates.hpp>

namespace ArborX
{

namespace Experimental
{
template <typename Values, typename Index>
struct AttachIndices
{
  Values _values;
};

// Make sure the default Index matches the default in PairValueIndex
template <typename Index = typename PairValueIndex<int>::index_type,
          typename Values = void>
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
private:
  using Self = ArborX::Experimental::AttachIndices<Values, Index>;
  using Access = AccessTraits<Values, ArborX::PrimitivesTag>;
  using value_type = ArborX::PairValueIndex<
      std::decay_t<Kokkos::detected_t<
          ArborX::Details::AccessTraitsGetArchetypeExpression, Access, Values>>,
      Index>;

public:
  using memory_space = typename Access::memory_space;

  KOKKOS_FUNCTION static auto size(Self const &self)
  {
    return Access::size(self._values);
  }
  KOKKOS_FUNCTION static auto get(Self const &self, int i)
  {
    return value_type{Access::get(self._values, i), Index(i)};
  }
};
template <typename Values, typename Index>
struct ArborX::AccessTraits<ArborX::Experimental::AttachIndices<Values, Index>,
                            ArborX::PredicatesTag>
{
private:
  using Self = ArborX::Experimental::AttachIndices<Values, Index>;
  using Access = AccessTraits<Values, ArborX::PredicatesTag>;

public:
  using memory_space = typename Access::memory_space;

  KOKKOS_FUNCTION static auto size(Self const &self)
  {
    return Access::size(self._values);
  }
  KOKKOS_FUNCTION static auto get(Self const &self, int i)
  {
    return attach(Access::get(self._values, i), Index(i));
  }
};

#endif
