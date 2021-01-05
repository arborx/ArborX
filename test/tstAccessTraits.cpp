/****************************************************************************
 * Copyright (c) 2017-2021 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <ArborX_AccessTraits.hpp>

#include <Kokkos_Core.hpp>

using ArborX::PredicatesTag;
using ArborX::PrimitivesTag;
using ArborX::Details::check_valid_access_traits;

struct NoAccessTraitsSpecialization
{
};
struct EmptySpecialization
{
};
struct InvalidMemorySpace
{
};
struct SizeMemberFunctionNotStatic
{
};
namespace ArborX
{
template <typename Tag>
struct AccessTraits<EmptySpecialization, Tag>
{
};
template <typename Tag>
struct AccessTraits<InvalidMemorySpace, Tag>
{
  using memory_space = void;
};
template <typename Tag>
struct AccessTraits<SizeMemberFunctionNotStatic, Tag>
{
  using memory_space = Kokkos::HostSpace;
  int size(SizeMemberFunctionNotStatic) { return 255; }
};
} // namespace ArborX

// Ensure legacy access traits are still valid
struct LegacyAccessTraits
{
};
namespace ArborX
{
namespace Traits
{
template <typename Tag>
struct Access<LegacyAccessTraits, Tag>
{
  using memory_space = Kokkos::HostSpace;
  KOKKOS_FUNCTION static int size(LegacyAccessTraits) { return 0; }
  KOKKOS_FUNCTION static Point get(LegacyAccessTraits, int) { return {}; }
};
} // namespace Traits
} // namespace ArborX

int main()
{
  Kokkos::View<ArborX::Point *> p;
  Kokkos::View<float **> v;
  check_valid_access_traits(PrimitivesTag{}, p);
  check_valid_access_traits(PrimitivesTag{}, v);

  using NearestPredicate = decltype(ArborX::nearest(ArborX::Point{}));
  Kokkos::View<NearestPredicate *> q;
  check_valid_access_traits(PredicatesTag{}, q);

  check_valid_access_traits(PrimitivesTag{}, LegacyAccessTraits{});

  // Uncomment to see error messages

  // check_valid_access_traits(PrimitivesTag{}, NoAccessTraitsSpecialization{});

  // check_valid_access_traits(PrimitivesTag{}, EmptySpecialization{});

  // check_valid_access_traits(PrimitivesTag{}, InvalidMemorySpace{});

  // check_valid_access_traits(PrimitivesTag{}, SizeMemberFunctionNotStatic{});
}
