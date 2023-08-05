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

#include <ArborX_AccessTraits.hpp>
#include <ArborX_HyperPoint.hpp>
#include <ArborX_Point.hpp>

#include <Kokkos_Core.hpp>

using ArborX::PredicatesTag;
using ArborX::PrimitivesTag;
using ArborX::Details::check_valid_access_traits;

// NOTE Let's not bother with __host__ __device__ annotations here

struct NoAccessTraitsSpecialization
{};

struct EmptySpecialization
{};
template <typename Tag>
struct ArborX::AccessTraits<EmptySpecialization, Tag>
{};

struct InvalidMemorySpace
{};
template <typename Tag>
struct ArborX::AccessTraits<InvalidMemorySpace, Tag>
{
  using memory_space = void;
};

struct SizeMemberFunctionNotStatic
{};
template <typename Tag>
struct ArborX::AccessTraits<SizeMemberFunctionNotStatic, Tag>
{
  using memory_space = Kokkos::HostSpace;
  int size(SizeMemberFunctionNotStatic) { return 255; }
};

// Ensure legacy access traits are still valid
struct LegacyAccessTraits
{};
template <typename Tag>
struct ArborX::Traits::Access<LegacyAccessTraits, Tag>
{
  using memory_space = Kokkos::HostSpace;
  static int size(LegacyAccessTraits) { return 0; }
  static Point get(LegacyAccessTraits, int) { return {}; }
};

void test_access_traits_compile_only()
{
  Kokkos::View<ArborX::Point *> p;
  Kokkos::View<float **> v;
  check_valid_access_traits(PrimitivesTag{}, p);
  check_valid_access_traits(PrimitivesTag{}, v);

  using NearestPredicate = decltype(ArborX::nearest(ArborX::Point{}));
  Kokkos::View<NearestPredicate *> q;
  check_valid_access_traits(PredicatesTag{}, q);

  // Uncomment to see error messages

  // check_valid_access_traits(PrimitivesTag{}, NoAccessTraitsSpecialization{});

  // check_valid_access_traits(PrimitivesTag{}, EmptySpecialization{});

  // check_valid_access_traits(PrimitivesTag{}, InvalidMemorySpace{});

  // check_valid_access_traits(PrimitivesTag{}, SizeMemberFunctionNotStatic{});

  // check_valid_access_traits(PrimitivesTag{}, LegacyAccessTraits{});
}

template <class V>
using deduce_point_t =
    decltype(ArborX::AccessTraits<V, ArborX::PrimitivesTag>::get(
        std::declval<V>(), 0));

void test_deduce_point_type_from_view()
{
  using GoodOlePoint = ArborX::Point;
  using ArborX::ExperimentalHyperGeometry::Point;
  static_assert(
      std::is_same_v<deduce_point_t<Kokkos::View<float **>>, GoodOlePoint>);
  static_assert(
      std::is_same_v<deduce_point_t<Kokkos::View<float *[3]>>, Point<3>>);
  static_assert(
      std::is_same_v<deduce_point_t<Kokkos::View<float *[2]>>, Point<2>>);
  static_assert(
      std::is_same_v<deduce_point_t<Kokkos::View<float *[5]>>, Point<5>>);
}
