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

#include <ArborX_Point.hpp>
#include <detail/ArborX_AccessTraits.hpp>
#include <detail/ArborX_AttachIndices.hpp>

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

template <class V, class Tag>
using deduce_type_t =
    decltype(ArborX::AccessTraits<V, Tag>::get(std::declval<V>(), 0));

void test_access_traits_compile_only()
{
  using Point = ArborX::Point<3>;

  Kokkos::View<Point *> p;
  Kokkos::View<float **> v;
  check_valid_access_traits(PrimitivesTag{}, p);
  check_valid_access_traits(PrimitivesTag{}, v);

  auto p_with_indices = ArborX::Experimental::attach_indices(p);
  check_valid_access_traits(PrimitivesTag{}, p_with_indices,
                            ArborX::Details::DoNotCheckGetReturnType());
  static_assert(
      std::is_same_v<deduce_type_t<decltype(p_with_indices), PrimitivesTag>,
                     ArborX::PairValueIndex<Point, unsigned>>);

  auto p_with_indices_long = ArborX::Experimental::attach_indices<long>(p);
  static_assert(std::is_same_v<
                deduce_type_t<decltype(p_with_indices_long), PrimitivesTag>,
                ArborX::PairValueIndex<Point, long>>);

  using NearestPredicate = decltype(ArborX::nearest(Point{}));
  Kokkos::View<NearestPredicate *> q;
  check_valid_access_traits(PredicatesTag{}, q);

  auto q_with_indices = ArborX::Experimental::attach_indices<long>(q);
  check_valid_access_traits(PredicatesTag{}, q_with_indices);
  using predicate = deduce_type_t<decltype(q_with_indices), PredicatesTag>;
  static_assert(
      std::is_same_v<
          std::decay_t<decltype(ArborX::getData(std::declval<predicate>()))>,
          long>);

  struct CustomIndex
  {
    char index;
    CustomIndex(int i) { index = i; }
  };
  auto q_with_custom_indices =
      ArborX::Experimental::attach_indices<CustomIndex>(q);
  check_valid_access_traits(PredicatesTag{}, q_with_custom_indices);
  using predicate_custom =
      deduce_type_t<decltype(q_with_custom_indices), PredicatesTag>;
  static_assert(std::is_same_v<std::decay_t<decltype(ArborX::getData(
                                   std::declval<predicate_custom>()))>,
                               CustomIndex>);

  // Uncomment to see error messages

  // check_valid_access_traits(PrimitivesTag{}, NoAccessTraitsSpecialization{});

  // check_valid_access_traits(PrimitivesTag{}, EmptySpecialization{});

  // check_valid_access_traits(PrimitivesTag{}, InvalidMemorySpace{});

  // check_valid_access_traits(PrimitivesTag{}, SizeMemberFunctionNotStatic{});
}

void test_deduce_point_type_from_view()
{
  using ArborX::Point;
  using ArborX::PrimitivesTag;
  static_assert(
      std::is_same_v<deduce_type_t<Kokkos::View<float **>, PrimitivesTag>,
                     Point<3>>);
  static_assert(
      std::is_same_v<deduce_type_t<Kokkos::View<float *[3]>, PrimitivesTag>,
                     Point<3>>);
  static_assert(
      std::is_same_v<deduce_type_t<Kokkos::View<float *[2]>, PrimitivesTag>,
                     Point<2>>);
  static_assert(
      std::is_same_v<deduce_type_t<Kokkos::View<float *[5]>, PrimitivesTag>,
                     Point<5>>);
}
