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

#include <ArborX.hpp>
#include <ArborX_Version.hpp>

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

struct ExcludeSelfCollision
{
  template <class Predicate, class OutputFunctor>
  KOKKOS_FUNCTION void operator()(Predicate const &predicate, int i,
                                  OutputFunctor const &out) const
  {
    int const j = getData(predicate);
    if (i != j)
    {
      out(i);
    }
  }
};

template <class BVH, class Counts, class Save>
struct CountAndSaveFirstCollision
{
  BVH _bvh;
  Counts _counts;
  Save _save;

  template <class Predicate>
  KOKKOS_FUNCTION void operator()(Predicate const &predicate, int i) const
  {
    int const j = getData(predicate);
    using ArborX::Details::HappyTreeFriends;
    if ((int)HappyTreeFriends::getLeafPermutationIndex(_bvh, i) != j)
    {
      if (++_counts(j) == 1)
      {
        _save(j) = i;
      }
    }
  }
};

template <class BVH, class Counts, class Offsets, class Indices>
struct CountAndFill
{
  BVH _bvh;
  Counts _counts;
  Offsets _offsets;
  Indices _indices;

  template <class Predicate>
  KOKKOS_FUNCTION void operator()(Predicate const &predicate, int i) const
  {
    int const j = getData(predicate);
    using ArborX::Details::HappyTreeFriends;
    i = HappyTreeFriends::getLeafPermutationIndex(_bvh, i);
    if (i != j)
    {
      _indices(_offsets(j) + _counts(j)++) = i;
    }
  }
};

template <class ExecutionSpace, class BVH, class Predicates, class Offsets,
          class Indices>
void query_with_restart(ExecutionSpace const &space, BVH const &bvh,
                        Predicates const &predicates, Offsets &offsets,
                        Indices &indices)
{
  Kokkos::Profiling::pushRegion("ArborX::Experimental::query_with_restart");
  Kokkos::Profiling::pushRegion("ArborX::Experimental::FirstPass::count");

  using MemorySpace = typename ExecutionSpace::memory_space;
  using Access = ArborX::AccessTraits<Predicates, ArborX::PredicatesTag>;
  auto const n = Access::size(predicates);

  Kokkos::View<int *, MemorySpace> counts(
      Kokkos::view_alloc("ArborX::Experimental::counts",
                         Kokkos::WithoutInitializing),
      n);
  Kokkos::deep_copy(space, counts, 0);
  Kokkos::View<int *, MemorySpace> restart(
      Kokkos::view_alloc("ArborX::Experimental::first_collision",
                         Kokkos::WithoutInitializing),
      n);
  ArborX::Details::restartRopeTraversal(
      space, bvh, predicates,
      CountAndSaveFirstCollision<BVH, decltype(counts), decltype(restart)>{
          bvh, counts, restart},
      ArborX::Details::StartFromRoot<BVH>{bvh});

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::pushRegion("ArborX::Experimental::SecondPass::fill");

  ArborX::reallocWithoutInitializing(offsets, n + 1);
  Kokkos::deep_copy(space, offsets, 0);
  Kokkos::deep_copy(space, Kokkos::subview(offsets, std::make_pair(0, (int)n)),
                    counts);
  ArborX::exclusivePrefixSum(space, offsets);
  auto const n_collisions = ArborX::lastElement(offsets);
  ArborX::reallocWithoutInitializing(indices, n_collisions);
  Kokkos::deep_copy(space, counts, 0);

  ArborX::Details::restartRopeTraversal(
      space, bvh, predicates,
      CountAndFill<BVH, decltype(counts), Offsets, Indices>{bvh, counts,
                                                            offsets, indices},
      restart);
  // ArborX::Details::StartFromRoot<BVH>{bvh});

  Kokkos::Profiling::popRegion();
  Kokkos::Profiling::popRegion();
}

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  std::cout << "ArborX version    : " << ArborX::version() << std::endl;
  std::cout << "ArborX hash       : " << ArborX::gitCommitHash() << std::endl;
  std::cout << "Kokkos version    : " << KokkosExt::version() << std::endl;

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
        // face-centered cubic arrangement of particles
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
          for (int d = 0; d < 3; ++d)
          {
            velocities(i, d) = generator.frand(0.f, 1.f);
          }
          random_pool.free_state(generator);
        });
  }
  // TODO scale velocities
  Kokkos::Profiling::popRegion();

  ArborX::BVH<MemorySpace> index(execution_space, particles);

  Kokkos::View<int *, MemorySpace> indices("Example::indices", 0);
  Kokkos::View<int *, MemorySpace> offsets("Example::offsets", 0);
  index.query(
      execution_space, Neighbors<MemorySpace>{particles, r},
      ExcludeSelfCollision{}, indices, offsets,
      ArborX::Experimental::TraversalPolicy().setPredicateSorting(false));

  query_with_restart(execution_space, index,
                     Neighbors<MemorySpace>{particles, r}, offsets, indices);

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
          // Typically, the neighbor search radius will be greater than the
          // cut-off distance, hence the if condition below.
          auto const cutoff_sq_ij =
              KokkosExt::ArithmeticTraits::infinity<decltype(rsq)>::value;
          if (rsq < cutoff_sq_ij)
          {
            auto const r2inv = static_cast<decltype(rsq)>(1) / rsq;
            auto const r6inv = r2inv * r2inv * r2inv;
            auto const lj1_ij = 1.f;
            auto const lj2_ij = 1.f;
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
          auto const cutoff_sq_ij =
              KokkosExt::ArithmeticTraits::infinity<decltype(rsq)>::value;
          if (rsq < cutoff_sq_ij)
          {
            auto const r2inv = static_cast<decltype(rsq)>(1) / rsq;
            auto const r6inv = r2inv * r2inv * r2inv;
            auto const lj1_ij = 1.f;
            auto const lj2_ij = 1.f;
            local_energy = .5f * r6inv * (.5f * lj1_ij * r6inv - lj2_ij) / 6.f;
          }
        }
      },
      potential_energy);

  Kokkos::parallel_for(
      "Example::update_particles_position_and_velocity",
      Kokkos::RangePolicy<ExecutionSpace>(execution_space, 0, n),
      KOKKOS_LAMBDA(int i) {
        auto const mass_i = 1.f;
        auto const dt_m = dt / mass_i;
        for (int d = 0; d < 3; ++d)
        {
          velocities(i, d) += dt_m * forces(i, d);
          particles(i)[d] += dt * velocities(i, d);
        }
      });
}
