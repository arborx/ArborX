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

#include "ArborX_EnableDeviceTypes.hpp" // ARBORX_DEVICE_TYPES
#include <ArborX_Point.hpp>
#include <ArborX_Sphere.hpp>
#include <algorithms/ArborX_Equals.hpp>
#include <detail/ArborX_PredicateHelpers.hpp>

#include <Kokkos_Core.hpp>

#include "BoostTest_CUDA_clang_workarounds.hpp"
#include <boost/test/unit_test.hpp>

#include <tuple>
#include <vector>

template <typename MemorySpace>
struct Placeholder
{
  using memory_space = MemorySpace;
  int n;
};

struct PlaceholderWithoutMemory
{
  int n;
};

template <typename MemorySpace>
struct ArborX::AccessTraits<Placeholder<MemorySpace>>
{
  using Self = Placeholder<MemorySpace>;
  using memory_space = MemorySpace;

  KOKKOS_FUNCTION static auto size(Self const &self) { return self.n; }
  KOKKOS_FUNCTION static auto get(Self const &, int i)
  {
    return ArborX::Point<2>{(float)i, (float)i};
  }
};

template <>
struct ArborX::AccessTraits<PlaceholderWithoutMemory>
{
  using Self = PlaceholderWithoutMemory;
  using memory_space = Kokkos::DefaultExecutionSpace::memory_space;

  KOKKOS_FUNCTION static auto size(Self const &self) { return self.n; }
  KOKKOS_FUNCTION static auto get(Self const &, int i)
  {
    return ArborX::Point<2>{(float)i, (float)i};
  }
};

struct IntersectsTag
{};
struct IntersectsWithRadiusTag
{};
struct NearestTag
{};

template <typename Tag, typename ExecutionSpace, typename Predicates,
          typename Data, typename... Args>
bool checkPredicates(Tag, ExecutionSpace const &space,
                     Predicates const &user_predicates, Data const &user_data,
                     Args... args)
{
  using namespace ArborX::Details;

  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  AccessValues<Data> data{user_data};
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  AccessValues<Predicates> predicates{user_predicates};

  // Check that the predicates of the same type
  static_assert(std::is_same_v<
                typename AccessValues<Predicates>::value_type::Tag,
                std::conditional_t<std::is_same_v<Tag, NearestTag>,
                                   NearestPredicateTag, SpatialPredicateTag>>);

  // Check that the predicates are of the same size
  if (predicates.size() != data.size())
    return false;

  int const n = data.size();

  // Check that the predicates have the same geometry
  int num_equal = 0;
  if constexpr (std::is_same_v<Tag, IntersectsTag>)
  {
    Kokkos::parallel_reduce(
        "Testing::check_predicates", Kokkos::RangePolicy(space, 0, n),
        KOKKOS_LAMBDA(int i, int &update) {
          update += equals(data(i), ArborX::getGeometry(predicates(i)));
        },
        num_equal);
  }
  else if constexpr (std::is_same_v<Tag, IntersectsWithRadiusTag>)
  {
    auto r = static_cast<float>(std::get<0>(std::make_tuple(args...)));
    Kokkos::parallel_reduce(
        "Testing::check_predicates", Kokkos::RangePolicy(space, 0, n),
        KOKKOS_LAMBDA(int i, int &update) {
          update += equals(ArborX::Sphere(data(i), r),
                           ArborX::getGeometry(predicates(i)));
        },
        num_equal);
  }
  else if constexpr (std::is_same_v<Tag, NearestTag>)
  {
    auto k = static_cast<int>(std::get<0>(std::make_tuple(args...)));
    Kokkos::parallel_reduce(
        "Testing::check_predicates", Kokkos::RangePolicy(space, 0, n),
        KOKKOS_LAMBDA(int i, int &update) {
          update += equals(data(i), ArborX::getGeometry(predicates(i))) &&
                    ArborX::getK(predicates(i)) == k;
        },
        num_equal);
  }

  return num_equal == n;
}

BOOST_AUTO_TEST_CASE_TEMPLATE(make_predicates, DeviceType, ARBORX_DEVICE_TYPES)
{
  using MemorySpace = typename DeviceType::memory_space;
  using ExecutionSpace = typename DeviceType::execution_space;

  using namespace ArborX::Experimental;

  using Point = ArborX::Point<2>;

  ExecutionSpace space;

  Kokkos::View<Point *, MemorySpace> points_view("Testing::points", 3);
  std::vector<Point> v = {Point{0, 0}, Point{1, 1}, Point{2, 2}};
  Kokkos::deep_copy(
      points_view,
      Kokkos::View<Point *, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
          v.data(), v.size()));

  Placeholder<MemorySpace> points_access{3};
  PlaceholderWithoutMemory points_access_nomem{3};

  BOOST_TEST(checkPredicates(IntersectsTag{}, space,
                             make_intersects(points_view), points_view));
  BOOST_TEST(checkPredicates(IntersectsTag{}, space,
                             make_intersects(points_access), points_access));
  BOOST_TEST(checkPredicates(IntersectsTag{}, space,
                             make_intersects(points_access_nomem),
                             points_access_nomem));

  float r = 1.f;
  BOOST_TEST(checkPredicates(IntersectsWithRadiusTag{}, space,
                             make_intersects(points_view, r), points_view, r));
  BOOST_TEST(checkPredicates(IntersectsWithRadiusTag{}, space,
                             make_intersects(points_access, r), points_access,
                             r));
  BOOST_TEST(checkPredicates(IntersectsWithRadiusTag{}, space,
                             make_intersects(points_access_nomem, r),
                             points_access_nomem, r));

  int const k = 3;
  BOOST_TEST(checkPredicates(NearestTag{}, space, make_nearest(points_view, k),
                             points_view, k));
  BOOST_TEST(checkPredicates(NearestTag{}, space,
                             make_nearest(points_access, k), points_access, k));
  BOOST_TEST(checkPredicates(NearestTag{}, space,
                             make_nearest(points_access_nomem, k),
                             points_access_nomem, k));
}
