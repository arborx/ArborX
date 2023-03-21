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

#ifndef ARBORX_TEST_CLOUD_HPP
#define ARBORX_TEST_CLOUD_HPP

#include <ArborX_Box.hpp>
#include <ArborX_Point.hpp>

#include <Kokkos_Random.hpp>

namespace ArborXTest
{

template <typename Geometry, typename ExecutionSpace>
Kokkos::View<Geometry *, ExecutionSpace>
make_random_cloud(ExecutionSpace const &space, int n, float Lx = 1.f,
                  float Ly = 1.f, float Lz = 1.f, int const seed = 0)
{
  static_assert(std::is_same_v<Geometry, ArborX::Point> ||
                std::is_same_v<Geometry, ArborX::Box>);

  // We divide n to avoid a large number of overlapping boxes
  auto const Ll = std::min({Lx, Ly, Lz}) / n;

  Kokkos::View<Geometry *, ExecutionSpace> cloud(
      Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                         "ArborXTest::cloud"),
      n);

  using RandomPool = Kokkos::Random_XorShift64_Pool<ExecutionSpace>;
  RandomPool random_pool(seed);
  Kokkos::parallel_for(
      "ArborXTest::generate_random_cloud",
      Kokkos::RangePolicy<ExecutionSpace>(space, 0, n), KOKKOS_LAMBDA(int i) {
        typename RandomPool::generator_type generator = random_pool.get_state();
        auto const x = generator.frand(0.f, Lx);
        auto const y = generator.frand(0.f, Ly);
        auto const z = generator.frand(0.f, Lz);

#ifdef KOKKOS_COMPILER_NVCC
        // Workaround NVCC error "An extended __host__ __device__ lambda cannot
        // first-capture variable in constexpr-if context"
        (void)cloud;
        (void)Ll;
#endif

        if constexpr (std::is_same_v<Geometry, ArborX::Point>)
        {
          cloud(i) = {x, y, z};
        }
        else
        {
          auto const length = generator.frand(0.f, Ll);
          cloud(i) = {{x, y, z}, {x + length, y + length, z + length}};
        }
        random_pool.free_state(generator);
      });
  space.fence();

  return cloud;
}

} // namespace ArborXTest

#endif
