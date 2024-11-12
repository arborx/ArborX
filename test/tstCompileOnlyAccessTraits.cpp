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

using ArborX::Details::check_valid_access_traits;
using ArborX::Details::CheckReturnTypeTag;

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

struct MissingSizeMemberFunction
{};
template <typename Tag>
struct ArborX::AccessTraits<MissingSizeMemberFunction, Tag>
{
  using memory_space = Kokkos::HostSpace;
};

struct SizeMemberFunctionNotStatic
{};
template <typename Tag>
struct ArborX::AccessTraits<SizeMemberFunctionNotStatic, Tag>
{
  using memory_space = Kokkos::HostSpace;
  int size(SizeMemberFunctionNotStatic) { return 255; }
};

struct MissingGetMemberFunction
{};
template <typename Tag>
struct ArborX::AccessTraits<MissingGetMemberFunction, Tag>
{
  using memory_space = Kokkos::HostSpace;
  static int size(MissingGetMemberFunction) { return 255; }
};

struct GetMemberFunctionNotStatic
{};
template <typename Tag>
struct ArborX::AccessTraits<GetMemberFunctionNotStatic, Tag>
{
  using memory_space = Kokkos::HostSpace;
  static int size(GetMemberFunctionNotStatic) { return 255; }
  ArborX::Point<3> get(GetMemberFunctionNotStatic, int) { return {}; }
};

struct GetMemberFunctionVoid
{};
template <typename Tag>
struct ArborX::AccessTraits<GetMemberFunctionVoid, Tag>
{
  using memory_space = Kokkos::HostSpace;
  static int size(GetMemberFunctionVoid) { return 255; }
  static void get(GetMemberFunctionVoid, int) {}
};

template <class V>
using deduce_type_t =
    decltype(ArborX::AccessTraits<V>::get(std::declval<V>(), 0));

void test_access_traits_compile_only()
{
  using Point = ArborX::Point<3>;

  Kokkos::View<Point *> p;
  Kokkos::View<float **> v;
  check_valid_access_traits(p);
  check_valid_access_traits(v);

  auto p_with_indices = ArborX::Experimental::attach_indices(p);
  check_valid_access_traits(p_with_indices);
  static_assert(std::is_same_v<deduce_type_t<decltype(p_with_indices)>,
                               ArborX::PairValueIndex<Point, unsigned>>);

  auto p_with_indices_long = ArborX::Experimental::attach_indices<long>(p);
  static_assert(std::is_same_v<deduce_type_t<decltype(p_with_indices_long)>,
                               ArborX::PairValueIndex<Point, long>>);

  using NearestPredicate = decltype(ArborX::nearest(Point{}));
  Kokkos::View<NearestPredicate *> q;
  check_valid_access_traits(q, CheckReturnTypeTag{});

  auto q_with_indices = ArborX::Experimental::attach_indices<long>(q);
  check_valid_access_traits(q_with_indices, CheckReturnTypeTag{});
  using predicate = deduce_type_t<decltype(q_with_indices)>;
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
  check_valid_access_traits(q_with_custom_indices, CheckReturnTypeTag{});
  using predicate_custom = deduce_type_t<decltype(q_with_custom_indices)>;
  static_assert(std::is_same_v<std::decay_t<decltype(ArborX::getData(
                                   std::declval<predicate_custom>()))>,
                               CustomIndex>);

  // Uncomment to see error messages

  // check_valid_access_traits(NoAccessTraitsSpecialization{});

  // check_valid_access_traits(EmptySpecialization{});

  // check_valid_access_traits(InvalidMemorySpace{});

  // check_valid_access_traits(MissingSizeMemberFunction{});

  // check_valid_access_traits(SizeMemberFunctionNotStatic{});

  // check_valid_access_traits(MissingGetMemberFunction{});

  // check_valid_access_traits(GetMemberFunctionNotStatic{});

  // check_valid_access_traits(GetMemberFunctionVoid{});
}

void test_deduce_point_type_from_view()
{
  using ArborX::Point;
  static_assert(
      std::is_same_v<deduce_type_t<Kokkos::View<float **>>, Point<3>>);
  static_assert(
      std::is_same_v<deduce_type_t<Kokkos::View<float *[3]>>, Point<3>>);
  static_assert(
      std::is_same_v<deduce_type_t<Kokkos::View<float *[2]>>, Point<2>>);
  static_assert(
      std::is_same_v<deduce_type_t<Kokkos::View<float *[5]>>, Point<5>>);
}
