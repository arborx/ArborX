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
#ifndef ARBORX_ATTACH_INDICES_HPP
#define ARBORX_ATTACH_INDICES_HPP

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
struct ArborX::AccessTraits<ArborX::Experimental::AttachIndices<Values, Index>>
{
private:
  using Self = ArborX::Experimental::AttachIndices<Values, Index>;
  using Access = AccessTraits<Values>;

public:
  using memory_space = typename Access::memory_space;

  KOKKOS_FUNCTION static auto size(Self const &self)
  {
    return Access::size(self._values);
  }
  KOKKOS_FUNCTION static auto get(Self const &self, int i)
  {
    using namespace ArborX::Details;

    using value_type = std::decay_t<
        Kokkos::detected_t<AccessTraitsGetArchetypeExpression, Access, Values>>;
    using PredicateTag =
        Kokkos::detected_t<PredicateTagArchetypeAlias, value_type>;

    if constexpr (is_valid_predicate_tag<PredicateTag>::value)
      return attach(Access::get(self._values, i), Index(i));
    else
      return PairValueIndex<value_type, Index>{Access::get(self._values, i),
                                               Index(i)};
  }
};

#endif
