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

#include <ArborX_Point.hpp>
#include <detail/ArborX_AccessTraits.hpp>
#include <detail/ArborX_AttachIndices.hpp>

#include <Kokkos_Core.hpp>

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

struct MissingPredicateTag
{};
template <typename Tag>
struct ArborX::AccessTraits<MissingPredicateTag, Tag>
{
  using memory_space = Kokkos::HostSpace;
  static int size(MissingPredicateTag) { return 255; }
  static auto get(MissingPredicateTag, int) { return 0; }
};

struct InvalidPredicateTag
{};
struct CustomTag
{
  using Tag = int;
};
template <typename Tag>
struct ArborX::AccessTraits<InvalidPredicateTag, Tag>
{
  using memory_space = Kokkos::HostSpace;
  static int size(InvalidPredicateTag) { return 255; }
  static auto get(InvalidPredicateTag, int) { return CustomTag{}; }
};

template <class V>
using deduce_type_t =
    decltype(ArborX::AccessTraits<V>::get(std::declval<V>(), 0));

void test_access_traits_compile_only()
{
  using Point = ArborX::Point<3>;

  static_assert(ArborX::Details::Concepts::AccessTraits<Kokkos::View<Point *>>);
  static_assert(
      ArborX::Details::Concepts::AccessTraits<Kokkos::View<float **>>);

  Kokkos::View<Point *> p;
  auto p_with_indices = ArborX::Experimental::attach_indices(p);
  static_assert(
      ArborX::Details::Concepts::AccessTraits<decltype(p_with_indices)>);
  static_assert(std::is_same_v<deduce_type_t<decltype(p_with_indices)>,
                               ArborX::PairValueIndex<Point, unsigned>>);

  auto p_with_indices_long = ArborX::Experimental::attach_indices<long>(p);
  static_assert(std::is_same_v<deduce_type_t<decltype(p_with_indices_long)>,
                               ArborX::PairValueIndex<Point, long>>);

  using NearestPredicate = decltype(ArborX::nearest(Point{}));
  Kokkos::View<NearestPredicate *> q;
  static_assert(ArborX::Details::Concepts::Predicates<decltype(q)>);

  auto q_with_indices = ArborX::Experimental::attach_indices<long>(q);
  using PredicatesWithIndices = decltype(q_with_indices);
  static_assert(ArborX::Details::Concepts::Predicates<PredicatesWithIndices>);
  using predicate = deduce_type_t<PredicatesWithIndices>;
  static_assert(
      std::is_same_v<
          std::decay_t<decltype(ArborX::getData(std::declval<predicate>()))>,
          long>);

  struct CustomIndex
  {
    char index;

    KOKKOS_FUNCTION
    CustomIndex(int i) { index = i; }
  };
  auto q_with_custom_indices =
      ArborX::Experimental::attach_indices<CustomIndex>(q);
  using PredicatesWithCustomIndices = decltype(q_with_custom_indices);
  static_assert(
      ArborX::Details::Concepts::Predicates<PredicatesWithCustomIndices>);
  using predicate_custom = deduce_type_t<PredicatesWithCustomIndices>;
  static_assert(std::is_same_v<std::decay_t<decltype(ArborX::getData(
                                   std::declval<predicate_custom>()))>,
                               CustomIndex>);

  static_assert(
      !ArborX::Details::Concepts::AccessTraits<NoAccessTraitsSpecialization>);
  static_assert(!ArborX::Details::Concepts::AccessTraits<EmptySpecialization>);
  static_assert(!ArborX::Details::Concepts::AccessTraits<InvalidMemorySpace>);
  static_assert(
      !ArborX::Details::Concepts::AccessTraits<MissingSizeMemberFunction>);
  static_assert(
      !ArborX::Details::Concepts::AccessTraits<SizeMemberFunctionNotStatic>);
  static_assert(
      !ArborX::Details::Concepts::AccessTraits<MissingGetMemberFunction>);
  static_assert(
      !ArborX::Details::Concepts::AccessTraits<GetMemberFunctionNotStatic>);
  static_assert(
      !ArborX::Details::Concepts::AccessTraits<GetMemberFunctionVoid>);
  static_assert(!ArborX::Details::Concepts::Predicates<MissingPredicateTag>);
  static_assert(!ArborX::Details::Concepts::Predicates<InvalidPredicateTag>);
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
