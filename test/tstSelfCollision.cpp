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

// clang-format off
#include "boost_ext/KokkosPairComparison.hpp"
#include "boost_ext/TupleComparison.hpp"
#include "boost_ext/CompressedStorageComparison.hpp"
// clang-format on

#include "ArborXTest_StdVectorToKokkosView.hpp"
#include "ArborX_EnableDeviceTypes.hpp" // ARBORX_DEVICE_TYPES
#include "ArborX_EnableViewComparison.hpp"
#include <ArborX_DetailsExpandHalfToFull.hpp>
#include <ArborX_DetailsSelfCollision.hpp>

#include <Kokkos_Random.hpp>

#include "BoostTest_CUDA_clang_workarounds.hpp"
#include <boost/test/unit_test.hpp>

namespace Test
{
using ArborXTest::toView;

template <class ExecutionSpace>
auto make_random_cloud(ExecutionSpace const &space, int n)
{
  Kokkos::View<ArborX::Point *, ExecutionSpace> points(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing, "Test::points"),
      n);
  using RandomPool = Kokkos::Random_XorShift64_Pool<ExecutionSpace>;
  RandomPool random_pool(5374857);

  Kokkos::parallel_for(
      "Test::generate_random_points",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n), KOKKOS_LAMBDA(int i) {
        typename RandomPool::generator_type generator = random_pool.get_state();
        auto const x = generator.frand(0.f, 1.f);
        auto const y = generator.frand(0.f, 1.f);
        auto const z = generator.frand(0.f, 1.f);
        points(i) = {x, y, z};
        random_pool.free_state(generator);
      });

  return points;
}

struct Filter
{
  template <class Predicate, class OutputFunctor>
  void operator()(Predicate const &predicate, int i,
                  OutputFunctor const &out) const
  {
    int const j = getData(predicate);
    if (i < j)
    {
      out(i);
    }
  }
};

template <class Points>
struct RadiusSearch
{
  Points points;
  float radius;
};

template <class Predicates, class IndexType = int>
struct AttachIndices
{
  Predicates predicates;
};
} // namespace Test

template <class Points>
struct ArborX::AccessTraits<Test::RadiusSearch<Points>, ArborX::PredicatesTag>
{
  using Access = AccessTraits<Points, PrimitivesTag>;
  using Self = Test::RadiusSearch<Points>;
  using memory_space = typename Access::memory_space;
  using size_type = decltype(Access::size(std::declval<Points const &>()));
  static KOKKOS_FUNCTION size_type size(Self const &x)
  {
    return Access::size(x.points);
  }
  static KOKKOS_FUNCTION auto get(Self const &x, size_type i)
  {
    return intersects(Sphere{Access::get(x.points, i), x.radius});
  }
};

template <class Predicates, class IndexType>
struct ArborX::AccessTraits<Test::AttachIndices<Predicates, IndexType>,
                            ArborX::PredicatesTag>
{
  using Access = AccessTraits<Predicates, PredicatesTag>;
  using Self = Test::AttachIndices<Predicates, IndexType>;
  using memory_space = typename Access::memory_space;
  using size_type = decltype(Access::size(std::declval<Predicates const &>()));
  static KOKKOS_FUNCTION size_type size(Self const &x)
  {
    return Access::size(x.predicates);
  }
  static KOKKOS_FUNCTION auto get(Self const &x, size_type i)
  {
    return attach(Access::get(x.predicates, i), static_cast<IndexType>(i));
  }
};

namespace Test
{

template <class ExecutionSpace>
auto expand(ExecutionSpace space, std::vector<int> const &offsets_host,
            std::vector<int> const &indices_host)
{
  auto offsets = toView<ExecutionSpace>(offsets_host, "Test::offsets");
  auto indices = toView<ExecutionSpace>(indices_host, "Test::indices");
  ArborX::Details::expandHalfToFull(space, offsets, indices);

  return make_compressed_storage(
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, offsets),
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, indices));
}

#define ARBORX_TEST_EXPAND_HALF_TO_FULL(exec_space, offsets_in, indices_in,    \
                                        offsets_out, indices_out)              \
  BOOST_TEST(Test::expand(exec_space, offsets_in, indices_in) ==               \
                 make_compressed_storage(offsets_out, indices_out),            \
             boost::test_tools::per_element())

} // namespace Test

BOOST_AUTO_TEST_CASE_TEMPLATE(self_collision_spatial, DeviceType,
                              ARBORX_DEVICE_TYPES)
{
  using MemorySpace = typename DeviceType::memory_space;
  using ExecutionSpace = typename DeviceType::execution_space;
  ExecutionSpace exec_space;

  ARBORX_TEST_EXPAND_HALF_TO_FULL(exec_space, (std::vector<int>{0}),
                                  (std::vector<int>{}), (std::vector<int>{0}),
                                  (std::vector<int>{}));
  auto points = Test::make_random_cloud(exec_space, 100);
  Kokkos::View<int *, ExecutionSpace> offsets("Test::offsets", 0);
  Kokkos::View<int *, ExecutionSpace> indices("Test::indices", 0);
  ArborX::Details::cabana_proxy(exec_space, points, 1.f, offsets, indices);
  ArborX::BoundingVolumeHierarchy<MemorySpace> bvh(exec_space, points);
  Test::RadiusSearch<decltype(points)> predicates{points, 1.f};
  bvh.query(exec_space, Test::AttachIndices<decltype(predicates)>{predicates},
            Test::Filter{}, indices, offsets);
}
