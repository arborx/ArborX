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

#ifndef ARBORX_SOD_HPP
#define ARBORX_SOD_HPP

#include <ArborX.hpp>
#include <ArborX_DetailsSOD.hpp>

namespace ArborX
{

template <typename MemorySpace>
struct InputData
{
  template <typename T>
  using View = Kokkos::View<T *, MemorySpace>;

  View<Point> particles;
  View<float> particle_masses;

  View<int64_t> fof_halo_tags;
  View<int> fof_halo_sizes;
  View<float> fof_halo_masses;
  View<Point> fof_halo_centers;

  InputData()
      : particles("particles", 0)
      , particle_masses("particle_masses", 0)
      , fof_halo_tags("fof_halo_tags", 0)
      , fof_halo_sizes("fof_halo_sizes", 0)
      , fof_halo_masses("fof_halo_masses", 0)
      , fof_halo_centers("fof_halo_centers", 0)
  {
  }
};

template <typename MemorySpace>
struct OutputData
{
  template <typename T>
  using View = Kokkos::View<T *, MemorySpace>;
  template <typename T>
  using BinView = Kokkos::View<T * [NUM_SOD_BINS], MemorySpace>;

  View<int64_t> fof_halo_tags;

  View<float> sod_halo_masses;
  View<int64_t> sod_halo_sizes;
  View<float> sod_halo_rdeltas;

  BinView<int> sod_halo_bin_ids;
  BinView<int> sod_halo_bin_counts;
  BinView<float> sod_halo_bin_masses;
  BinView<float> sod_halo_bin_outer_radii;
  BinView<float> sod_halo_bin_rhos;
  BinView<float> sod_halo_bin_rho_ratios;
  BinView<float> sod_halo_bin_radial_velocities;

  OutputData()
      : fof_halo_tags("fof_halo_tags", 0)
      , sod_halo_masses("sod_halo_masses", 0)
      , sod_halo_sizes("sod_halo_sizes", 0)
      , sod_halo_rdeltas("sod_halo_rdeltas", 0)
      , sod_halo_bin_ids("sod_halo_bin_ids", 0)
      , sod_halo_bin_counts("sod_halo_bin_counts", 0)
      , sod_halo_bin_masses("sod_halo_bin_masses", 0)
      , sod_halo_bin_outer_radii("sod_halo_bin_outer_radii", 0)
      , sod_halo_bin_rhos("sod_halo_bin_rhos", 0)
      , sod_halo_bin_rho_ratios("sod_halo_bin_rho_ratios", 0)
      , sod_halo_bin_radial_velocities("sod_hlo_bin_radial_velocities", 0)
  {
  }
};

template <typename MemorySpace>
struct Spheres
{
  Kokkos::View<Point *, MemorySpace> _centers;
  Kokkos::View<float *, MemorySpace> _radii;
};

template <typename MemorySpace>
struct AccessTraits<Spheres<MemorySpace>, PrimitivesTag>
{
  using memory_space = MemorySpace;

  KOKKOS_FUNCTION static std::size_t size(const Spheres<MemorySpace> &spheres)
  {
    return spheres._centers.extent(0);
  }
  KOKKOS_FUNCTION static Box get(Spheres<MemorySpace> const &spheres,
                                 std::size_t const i)
  {
    auto const &c = spheres._centers(i);
    auto const r = spheres._radii(i);
    return {{c[0] - r, c[1] - r, c[2] - r}, {c[0] + r, c[1] + r, c[2] + r}};
  }
};

template <typename Particles>
struct ParticlesWrapper
{
  Particles _particles;
};

template <typename Particles>
struct AccessTraits<ParticlesWrapper<Particles>, PredicatesTag>
{
  using ParticlesAccess = AccessTraits<Particles, PrimitivesTag>;

  using memory_space = typename ParticlesAccess::memory_space;
  using Predicates = ParticlesWrapper<Particles>;

  static KOKKOS_FUNCTION size_t size(Predicates const &w)
  {
    return ParticlesAccess::size(w._particles);
  }
  static KOKKOS_FUNCTION auto get(Predicates const &w, size_t i)
  {
    return attach(intersects(ParticlesAccess::get(w._particles, i)), (int)i);
  }
};

template <typename ExecutionSpace>
void sod(ExecutionSpace const &exec_space,
         InputData<Kokkos::HostSpace> const &in,
         OutputData<Kokkos::HostSpace> &out)
{
  using MemorySpace = typename ExecutionSpace::memory_space;
  using HostExecutionSpace = Kokkos::DefaultHostExecutionSpace;

  HostExecutionSpace host_space;

  auto const num_halos = in.fof_halo_tags.extent_int(0);

  // Transfer data to device
  auto particles =
      Kokkos::create_mirror_view_and_copy(exec_space, in.particles);
  auto particle_masses =
      Kokkos::create_mirror_view_and_copy(exec_space, in.particle_masses);
  auto fof_halo_centers =
      Kokkos::create_mirror_view_and_copy(exec_space, in.fof_halo_centers);
  auto fof_halo_masses =
      Kokkos::create_mirror_view_and_copy(exec_space, in.fof_halo_masses);

  using Particles = decltype(particles);

  // HACC constants
  float constexpr RHO = 2.77536627e11;

  // rho = RHO * Efact*Efact * a*a
  // At redshift = 0, the factors are trivial:
  //   a = 1, Efact = 1,
  // so rho = RHO.

  // Do not sort for now, so as to not allocate additional memory, which would
  // take 8*n bytes (4 for Morton index, 4 for permutation index). This will
  // results in running out of memory for the hardest HACC problem on V100.
  bool const sort_predicates = false;

  // Compute R_min and R_max radii for every FOF halo
  float r_min;
  Kokkos::View<float *, MemorySpace> r_max;
  std::tie(r_min, r_max) =
      Details::computeSODRadii(exec_space, fof_halo_masses);

  // Construct the search index based on spheres (FOF centers with
  // corresponding R_max)
  BVH<MemorySpace> bvh(exec_space,
                       Spheres<MemorySpace>{fof_halo_centers, r_max});

  // Compute bin outer radii
  auto sod_halo_bin_outer_radii =
      Details::computeSODBinRadii(exec_space, r_min, r_max);

  // Step 2: compute some profiles (mass, count, avg radius);
  Kokkos::View<int * [NUM_SOD_BINS], MemorySpace> sod_halo_bin_counts(
      "ArborX::SOD::sod_halo_bin_counts", num_halos);
  // NOTE: we will accumulate float quantities into double in order to
  // avoid loss of precision, which will occur once we start adding small
  // quantities to large
  Kokkos::View<double * [NUM_SOD_BINS], MemorySpace> sod_halo_bin_masses(
      "ArborX::SOD::sod_halo_bin_masses", num_halos);
  Kokkos::View<double * [NUM_SOD_BINS], MemorySpace> sod_halo_bin_avg_radii(
      "ArborX::SOD::sod_halo_bin_avg_radii", num_halos);
  bvh.query(
      exec_space, ParticlesWrapper<Particles>{particles},
      Details::BinAccumulator<MemorySpace, Particles>{
          particles, particle_masses, sod_halo_bin_counts, sod_halo_bin_masses,
          sod_halo_bin_avg_radii, fof_halo_centers, r_min, r_max},
      Experimental::TraversalPolicy().setPredicateSorting(sort_predicates));
  Kokkos::parallel_for(
      "ArborX::SOD::normalize_avg_radii",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_halos),
      KOKKOS_LAMBDA(int halo_index) {
        for (int bin_id = 0; bin_id < NUM_SOD_BINS; ++bin_id)
          sod_halo_bin_avg_radii(halo_index, bin_id) /=
              sod_halo_bin_counts(halo_index, bin_id);
      });

  // Compute rhos and rho ratios
  Kokkos::View<float * [NUM_SOD_BINS], MemorySpace> sod_halo_bin_rhos;
  Kokkos::View<float * [NUM_SOD_BINS], MemorySpace> sod_halo_bin_rho_ratios;
  std::tie(sod_halo_bin_rhos, sod_halo_bin_rho_ratios) =
      Details::computeSODRhos(exec_space, RHO, sod_halo_bin_masses,
                              sod_halo_bin_avg_radii);

  // Figure out critical bins
  float const DELTA = 200;
  auto critical_bin_ids = Details::computeSODCriticalBins(
      exec_space, RHO, DELTA, sod_halo_bin_masses, sod_halo_bin_counts,
      sod_halo_bin_outer_radii);

  // Compute offsets for storing particles in critical bins
  Kokkos::View<int *, MemorySpace> critical_bin_offsets(
      "ArborX::SOD::critical_bin_offsets", num_halos + 1);
  Kokkos::parallel_for(
      "compute_critical_bins",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_halos),
      KOKKOS_LAMBDA(int halo_index) {
        critical_bin_offsets(halo_index) =
            sod_halo_bin_counts(halo_index, critical_bin_ids(halo_index));
      });
  exclusivePrefixSum(exec_space, critical_bin_offsets);

  auto num_critical_bin_particles = lastElement(critical_bin_offsets);
  printf("#particles in critical bins: %d\n", num_critical_bin_particles);

  // Find particles in critical bins for each halo
  Kokkos::View<int *, MemorySpace> critical_bin_indices(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "ArborX::SOD::critical_bin_indices"),
      num_critical_bin_particles);
  Kokkos::View<float *, MemorySpace> critical_bin_distances_augmented(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "ArborX::SOD::critical_bin_distances_augmented"),
      num_critical_bin_particles);
  auto offsets = clone(critical_bin_offsets);
  bvh.query(
      exec_space, ParticlesWrapper<Particles>{particles},
      Details::CriticalBinParticles<MemorySpace, Particles>{
          particles, offsets, critical_bin_indices,
          critical_bin_distances_augmented, critical_bin_ids, fof_halo_centers,
          sod_halo_bin_outer_radii, r_min, r_max},
      Experimental::TraversalPolicy().setPredicateSorting(sort_predicates));

  // Compute R_delta
  out.sod_halo_rdeltas = Details::computeSODRdeltas(
      exec_space, DELTA, RHO, in.particle_masses, sod_halo_bin_masses,
      sod_halo_bin_outer_radii, critical_bin_ids, critical_bin_offsets,
      critical_bin_indices, critical_bin_distances_augmented);

  // Copy required things to host
  auto sod_halo_bin_outer_radii_host =
      Kokkos::create_mirror_view_and_copy(host_space, sod_halo_bin_outer_radii);
  Kokkos::resize(out.sod_halo_bin_outer_radii, num_halos);
  Kokkos::deep_copy(out.sod_halo_bin_outer_radii,
                    sod_halo_bin_outer_radii_host);

  auto sod_halo_bin_counts_host =
      Kokkos::create_mirror_view_and_copy(host_space, sod_halo_bin_counts);
  Kokkos::resize(out.sod_halo_bin_counts, num_halos);
  Kokkos::deep_copy(out.sod_halo_bin_counts, sod_halo_bin_counts_host);

  auto sod_halo_bin_masses_host =
      Kokkos::create_mirror_view_and_copy(host_space, sod_halo_bin_masses);
  Kokkos::resize(out.sod_halo_bin_masses, num_halos);
  Kokkos::parallel_for(
      "copy_bin_masses",
      Kokkos::RangePolicy<HostExecutionSpace>(host_space, 0, num_halos),
      KOKKOS_LAMBDA(int halo_index) {
        // double -> float conversion
        for (int bin_id = 0; bin_id < NUM_SOD_BINS; ++bin_id)
          out.sod_halo_bin_masses(halo_index, bin_id) =
              sod_halo_bin_masses_host(halo_index, bin_id);
      });

  auto sod_halo_bin_rhos_host =
      Kokkos::create_mirror_view_and_copy(host_space, sod_halo_bin_rhos);
  Kokkos::resize(out.sod_halo_bin_rhos, num_halos);
  Kokkos::deep_copy(out.sod_halo_bin_rhos, sod_halo_bin_rhos_host);

  auto sod_halo_bin_rho_ratios_host =
      Kokkos::create_mirror_view_and_copy(host_space, sod_halo_bin_rho_ratios);
  Kokkos::resize(out.sod_halo_bin_rho_ratios, num_halos);
  Kokkos::deep_copy(out.sod_halo_bin_rho_ratios, sod_halo_bin_rho_ratios_host);

  // Free GPU space
  Kokkos::resize(particle_masses, 0);
  Kokkos::resize(fof_halo_masses, 0);
  Kokkos::resize(sod_halo_bin_outer_radii, 0);
  Kokkos::resize(sod_halo_bin_counts, 0);
  Kokkos::resize(sod_halo_bin_masses, 0);
  Kokkos::resize(sod_halo_bin_rhos, 0);
  Kokkos::resize(sod_halo_bin_rho_ratios, 0);

  auto sod_halo_rdeltas =
      Kokkos::create_mirror_view_and_copy(exec_space, out.sod_halo_rdeltas);

#if 0
  // Check overlaps
  auto const n = in.particles.extent_int(0);
  Kokkos::View<int *, MemorySpace> counts("counts", n);
  bvh.query(
      exec_space, ParticlesWrapper<Particles>{particles},
      Details::OverlapCount<MemorySpace, Particles>{particles, counts, fof_halo_centers,
                                           sod_halo_rdeltas},
      Experimental::TraversalPolicy().setPredicateSorting(sort_predicates));

  // Compute some statistics
  printf("Stats:\n");

  int num_inside = 0;
  Kokkos::parallel_reduce("a",
                          Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
                          KOKKOS_LAMBDA(int i, int &update) {
                            if (counts(i) > 0)
                              ++update;
                          },
                          num_inside);
  printf("  #particles inside spheres: %d/%d [%.2f]\n", num_inside, n,
         (100.f * num_inside) / n);

  int num_inside_multiple = 0;
  Kokkos::parallel_reduce("b",
                          Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
                          KOKKOS_LAMBDA(int i, int &update) {
                            if (counts(i) > 1)
                              ++update;
                          },
                          num_inside_multiple);
  printf("  #particles in multiple: %d\n", num_inside_multiple);

  if (num_inside_multiple > 0)
  {
    int max_multiple = 0;
    Kokkos::parallel_reduce(
        "c", Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
        KOKKOS_LAMBDA(int i, int &update) {
          if (counts(i) > update)
            update = counts(i);
        },
        Kokkos::Max<int>(max_multiple));
    if (num_inside_multiple > 0)
      printf("  max number of owners: %d\n", max_multiple);
  }
#endif
}

} // namespace ArborX

#endif
