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

#include <ArborX.hpp>

#include <Kokkos_Random.hpp>

template <class MemorySpace>
struct Neighbors
{
  Kokkos::View<ArborX::Point *, MemorySpace> _particles;
  float _radius;
};

template <class MemorySpace>
struct ArborX::AccessTraits<Neighbors<MemorySpace>, ArborX::PredicatesTag>
{
  using memory_space = MemorySpace;
  using size_type = std::size_t;
  static KOKKOS_FUNCTION size_type size(Neighbors<MemorySpace> const &x)
  {
    return x._particles.extent(0);
  }
  static KOKKOS_FUNCTION auto get(Neighbors<MemorySpace> const &x, size_type i)
  {
    return attach(intersects(Sphere{x._particles(i), x._radius}), (int)i);
  }
};

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = ExecutionSpace::memory_space;

  ExecutionSpace execution_space{};

  // face-centered cubic lattice parameters
  float const dx = 1.7f;
  float const dy = 1.7f;
  float const dz = 1.7f;
  int const nx = 100;
  int const ny = 100;
  int const nz = 100;
  int const n = 4 * nx * ny * nz;

  auto const dt = 5e-3f; // time step

  float const r = 3.f; // cut-off radius

  Kokkos::Profiling::pushRegion("Example::setup");
  Kokkos::View<ArborX::Point *, MemorySpace> particles(
      Kokkos::view_alloc("Example::points", Kokkos::WithoutInitializing), n);
  Kokkos::parallel_for(
      "Example::make_particles",
      Kokkos::MDRangePolicy<Kokkos::Rank<3>, ExecutionSpace>(
          execution_space, {0, 0, 0}, {nx, ny, nz}),
      KOKKOS_LAMBDA(int i, int j, int k) {
        int const id = i * ny * nz + j * nz + k;
        // fcc lattice
        particles[4 * id + 0] = {i * dx + .0f, j * dy + .0f, k * dz + .0f};
        particles[4 * id + 1] = {i * dx + .0f, j * dy + .5f, k * dz + .5f};
        particles[4 * id + 2] = {i * dx + .5f, j * dy + .0f, k * dz + .5f};
        particles[4 * id + 3] = {i * dx + .5f, j * dy + .5f, k * dz + .0f};
      });

  Kokkos::View<float * [3], MemorySpace> velocities(
      Kokkos::view_alloc("Example::velocities", Kokkos::WithoutInitializing),
      n);
  { // scope so that random number generation resources are released
    using RandomPool = Kokkos::Random_XorShift64_Pool<ExecutionSpace>;
    RandomPool random_pool(5374857);

    Kokkos::parallel_for(
        "Example::assign_velocities",
        Kokkos::RangePolicy<ExecutionSpace>(execution_space, 0, n),
        KOKKOS_LAMBDA(int i) {
          RandomPool::generator_type generator = random_pool.get_state();
          velocities(i, 0) = generator.frand(0.f, 1.f);
          velocities(i, 1) = generator.frand(0.f, 1.f);
          velocities(i, 2) = generator.frand(0.f, 1.f);
          random_pool.free_state(generator);
        });
  }
  // TODO scale velocities
  Kokkos::Profiling::popRegion();

  ArborX::BVH<MemorySpace> index(execution_space, particles);

  Kokkos::View<int *, MemorySpace> indices("Example::indices", 0);
  Kokkos::View<int *, MemorySpace> offsets("Example::offsets", 0);
  index.query(execution_space, Neighbors<MemorySpace>{particles, r}, indices,
              offsets);

  Kokkos::View<float * [3], MemorySpace> forces("Example::forces", n);
  Kokkos::parallel_for(
      "Example::compute_forces",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space, 0, n),
      KOKKOS_LAMBDA(int i) {
        auto const x_i = particles(i)[0];
        auto const y_i = particles(i)[1];
        auto const z_i = particles(i)[2];
        float fxi = 0.f;
        float fyi = 0.f;
        float fzi = 0.f;
        for (int j = offsets(i); j < offsets(i + 1); ++j)
        {
          auto const dx = x_i - particles(indices(j))[0];
          auto const dy = y_i - particles(indices(j))[1];
          auto const dz = z_i - particles(indices(j))[2];
          auto const rsq = dx * dx + dy * dy + dz * dz;
          auto const cutoff_sq_ij = KokkosExt::ArithmeticTraits::infinity<
              std::remove_cv_t<decltype(rsq)>>::value; // FIXME
          if (rsq < cutoff_sq_ij)
          {
            auto const r2inv = static_cast<decltype(rsq)>(1) / rsq;
            auto const r6inv = r2inv * r2inv * r2inv;
            auto const lj1_ij = 1.f; // FIXME
            auto const lj2_ij = 1.f; // FIXME
            auto const fij = (r6inv * (lj1_ij * r6inv - lj2_ij)) * r2inv;
            fxi += dx * fij;
            fyi += dy * fij;
            fzi += dz * fij;
          }
        }
        forces(i, 0) += fxi;
        forces(i, 1) += fyi;
        forces(i, 2) += fzi;
      });

  float potential_energy;
  Kokkos::parallel_reduce(
      "Example::compute_potential_energy",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space, 0, n),
      KOKKOS_LAMBDA(int i, float &local_energy) {
        auto const x_i = particles(i)[0];
        auto const y_i = particles(i)[1];
        auto const z_i = particles(i)[2];
        for (int j = offsets(i); j < offsets(i + 1); ++j)
        {
          auto const dx = x_i - particles(indices(j))[0];
          auto const dy = y_i - particles(indices(j))[1];
          auto const dz = z_i - particles(indices(j))[2];
          auto const rsq = dx * dx + dy * dy + dz * dz;
          auto const cutoff_sq_ij = KokkosExt::ArithmeticTraits::infinity<
              std::remove_cv_t<decltype(rsq)>>::value; // FIXME
          if (rsq < cutoff_sq_ij)
          {
            auto const r2inv = static_cast<decltype(rsq)>(1) / rsq;
            auto const r6inv = r2inv * r2inv * r2inv;
            auto const lj1_ij = 1.f; // FIXME
            auto const lj2_ij = 1.f; // FIXME
            local_energy = .5f * r6inv * (.5f * lj1_ij * r6inv - lj2_ij) / 6.f;
          }
        }
      },
      potential_energy);

  Kokkos::parallel_for(
      "Example::update_particles_position_and_velocity",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space, 0, n),
      KOKKOS_LAMBDA(int i) {
        auto const mass_i = 1.f; // FIXME
        auto const dt_m = dt / mass_i;
        velocities(i, 0) += dt_m * forces(i, 0);
        velocities(i, 1) += dt_m * forces(i, 1);
        velocities(i, 2) += dt_m * forces(i, 2);
        particles(i)[0] += dt * velocities(i, 0);
        particles(i)[1] += dt * velocities(i, 1);
        particles(i)[2] += dt * velocities(i, 2);
      });
}
