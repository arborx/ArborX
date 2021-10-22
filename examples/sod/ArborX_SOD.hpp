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
struct SODOutputData
{
  template <typename T>
  using View = Kokkos::View<T *, MemorySpace>;
  template <typename T>
  using BinView = Kokkos::View<T **, MemorySpace>;

  View<float> sod_halo_masses;
  View<float> sod_halo_rdeltas;

  BinView<int> sod_halo_bin_ids;
  BinView<int> sod_halo_bin_counts;
  BinView<float> sod_halo_bin_masses;
  BinView<float> sod_halo_bin_outer_radii;
  BinView<float> sod_halo_bin_rhos;
  BinView<float> sod_halo_bin_rho_ratios;
  BinView<float> sod_halo_bin_radial_velocities;

  View<int> sod_particles_offsets;
  View<int> sod_particles_indices;

  SODOutputData()
      : sod_halo_masses("sod_halo_masses", 0)
      , sod_halo_rdeltas("sod_halo_rdeltas", 0)
      , sod_halo_bin_ids("sod_halo_bin_ids", 0, 0)
      , sod_halo_bin_counts("sod_halo_bin_counts", 0, 0)
      , sod_halo_bin_masses("sod_halo_bin_masses", 0, 0)
      , sod_halo_bin_outer_radii("sod_halo_bin_outer_radii", 0, 0)
      , sod_halo_bin_rhos("sod_halo_bin_rhos", 0, 0)
      , sod_halo_bin_rho_ratios("sod_halo_bin_rho_ratios", 0, 0)
      , sod_halo_bin_radial_velocities("sod_hlo_bin_radial_velocities", 0, 0)
  {
  }
};

namespace SOD
{

template <typename MemorySpace>
struct Spheres
{
  Kokkos::View<Point *, MemorySpace> _centers;
  Kokkos::View<float *, MemorySpace> _radii;
};

template <typename Particles>
struct ParticlesWrapper
{
  Particles _particles;
};
} // namespace SOD

template <typename MemorySpace>
struct AccessTraits<SOD::Spheres<MemorySpace>, PrimitivesTag>
{
  using memory_space = MemorySpace;

  KOKKOS_FUNCTION static std::size_t
  size(const SOD::Spheres<MemorySpace> &spheres)
  {
    return spheres._centers.extent(0);
  }
  KOKKOS_FUNCTION static Box get(SOD::Spheres<MemorySpace> const &spheres,
                                 std::size_t const i)
  {
    auto const &c = spheres._centers(i);
    auto const r = spheres._radii(i);
    return {{c[0] - r, c[1] - r, c[2] - r}, {c[0] + r, c[1] + r, c[2] + r}};
  }
};

template <typename Particles>
struct AccessTraits<SOD::ParticlesWrapper<Particles>, PredicatesTag>
{
  using ParticlesAccess = AccessTraits<Particles, PrimitivesTag>;

  using memory_space = typename ParticlesAccess::memory_space;
  using Predicates = SOD::ParticlesWrapper<Particles>;

  static KOKKOS_FUNCTION size_t size(Predicates const &w)
  {
    return ParticlesAccess::size(w._particles);
  }
  static KOKKOS_FUNCTION auto get(Predicates const &w, size_t i)
  {
    return attach(intersects(ParticlesAccess::get(w._particles, i)), (int)i);
  }
};

namespace SOD
{

template <typename ExecutionSpace, typename MemorySpace, typename Particles,
          typename ParticleMasses, typename FOFHaloCenters,
          typename FOFHaloMasses>
void sodCore(ExecutionSpace const &exec_space, Particles &particles,
             ParticleMasses &particle_masses, FOFHaloCenters &fof_halo_centers,
             FOFHaloMasses &fof_halo_masses, SODOutputData<MemorySpace> &out,
             Parameters const &params)
{
  static_assert(
      KokkosExt::is_accessible_from<MemorySpace, ExecutionSpace>::value, "");
  static_assert(std::is_same<typename Particles::memory_space, MemorySpace>{},
                "");
  static_assert(
      std::is_same<typename ParticleMasses::memory_space, MemorySpace>{}, "");
  static_assert(
      std::is_same<typename FOFHaloCenters::memory_space, MemorySpace>{}, "");
  static_assert(
      std::is_same<typename FOFHaloMasses::memory_space, MemorySpace>{}, "");

  auto const num_halos = fof_halo_centers.extent(0);
  auto const num_sod_bins = params._num_sod_bins;

  // Do not sort for now, so as to not allocate additional memory, which would
  // take 8*n bytes (4 for Morton index, 4 for permutation index). This will
  // results in running out of memory for the hardest HACC problem on V100.
  bool const sort_predicates = false;

  // Compute R_min and R_max radii for every FOF halo
  float r_min;
  Kokkos::View<float *, MemorySpace> r_max;
  std::tie(r_min, r_max) = computeSODRadii(exec_space, params, fof_halo_masses);
  Kokkos::resize(fof_halo_masses, 0); // free as not used afterwards

  // Construct the search index based on spheres (FOF centers with
  // corresponding R_max)
  BVH<MemorySpace> bvh(exec_space,
                       Spheres<MemorySpace>{fof_halo_centers, r_max});

  // Compute bin outer radii
  out.sod_halo_bin_outer_radii =
      computeSODBinRadii(exec_space, r_min, r_max, num_sod_bins);

  // Step 2: compute some profiles (mass, count, avg radius);
  // NOTE: we will accumulate float quantities into double in order to
  // avoid loss of precision, which will occur once we start adding small
  // quantities to large
  Kokkos::View<double **, MemorySpace> sod_halo_bin_masses(
      "ArborX::SOD::sod_halo_bin_masses", num_halos, num_sod_bins);
  Kokkos::View<double **, MemorySpace> sod_halo_bin_avg_radii(
      "ArborX::SOD::sod_halo_bin_avg_radii", num_halos, num_sod_bins);
  Kokkos::resize(out.sod_halo_bin_counts, num_halos, num_sod_bins);
  bvh.query(
      exec_space, ParticlesWrapper<Particles>{particles},
      BinAccumulator<MemorySpace, Particles>{
          particles, particle_masses, out.sod_halo_bin_counts,
          sod_halo_bin_masses, sod_halo_bin_avg_radii, fof_halo_centers, r_min,
          r_max},
      Experimental::TraversalPolicy().setPredicateSorting(sort_predicates));
  Kokkos::parallel_for(
      "ArborX::SOD::normalize_avg_radii",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_halos),
      KOKKOS_LAMBDA(int halo_index) {
        for (int bin_id = 0; bin_id < num_sod_bins; ++bin_id)
          sod_halo_bin_avg_radii(halo_index, bin_id) /=
              out.sod_halo_bin_counts(halo_index, bin_id);
      });

  Kokkos::resize(out.sod_halo_bin_masses, num_halos, num_sod_bins);
  Kokkos::parallel_for(
      "ArborX::SOD::copy_bin_masses",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_halos),
      KOKKOS_LAMBDA(int halo_index) {
        // double -> float conversion
        for (int bin_id = 0; bin_id < num_sod_bins; ++bin_id)
          out.sod_halo_bin_masses(halo_index, bin_id) =
              sod_halo_bin_masses(halo_index, bin_id);
      });

  // Compute rhos and rho ratios
  std::tie(out.sod_halo_bin_rhos, out.sod_halo_bin_rho_ratios) = computeSODRhos(
      exec_space, params, sod_halo_bin_masses, sod_halo_bin_avg_radii);
  Kokkos::resize(sod_halo_bin_avg_radii, 0, 0); // free as not used aftewards

  // Figure out critical bins
  auto critical_bin_ids = computeSODCriticalBins(
      exec_space, params, sod_halo_bin_masses, out.sod_halo_bin_counts,
      out.sod_halo_bin_outer_radii);

  Kokkos::Profiling::pushRegion("ArborX::SOD::compute_critical_bin_particles");

  // Compute offsets for storing particles in critical bins
  Kokkos::View<int *, MemorySpace> critical_bin_offsets(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "ArborX::SOD::critical_bin_offsets"),
      num_halos + 1);
  Kokkos::parallel_for(
      "compute_critical_bins",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_halos),
      KOKKOS_LAMBDA(int halo_index) {
        critical_bin_offsets(halo_index) =
            out.sod_halo_bin_counts(halo_index, critical_bin_ids(halo_index));
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
  {
    auto offsets = clone(critical_bin_offsets);
    bvh.query(
        exec_space, ParticlesWrapper<Particles>{particles},
        CriticalBinParticles<MemorySpace, Particles>{
            particles, offsets, critical_bin_indices,
            critical_bin_distances_augmented, critical_bin_ids,
            fof_halo_centers, out.sod_halo_bin_outer_radii, r_min, r_max},
        Experimental::TraversalPolicy().setPredicateSorting(sort_predicates));
  }

  Kokkos::Profiling::popRegion();

  {
    // Permute found particles within a critical bin of each halo based on their
    // distance to the center
    Kokkos::Profiling::pushRegion(
        "ArborX::SOD::permute_critical_bin_particles");
    auto permute =
        Details::sortObjects(exec_space, critical_bin_distances_augmented);
    auto critical_bin_indices_clone = clone(critical_bin_indices);
    Kokkos::parallel_for("ArborX::SOD::apply_permutation",
                         Kokkos::RangePolicy<ExecutionSpace>(
                             exec_space, 0, num_critical_bin_particles),
                         KOKKOS_LAMBDA(int i) {
                           critical_bin_indices(i) =
                               critical_bin_indices_clone(permute(i));
                         });
    Kokkos::Profiling::popRegion();
  }

  // Compute R_delta
  Kokkos::View<int *, MemorySpace> sod_halo_rdeltas_index(
      "ArborX::SOD::r_delta_index", 0);
  std::tie(out.sod_halo_rdeltas, sod_halo_rdeltas_index) = computeSODRdeltas(
      exec_space, params, particle_masses, sod_halo_bin_masses,
      out.sod_halo_bin_outer_radii, critical_bin_ids, critical_bin_offsets,
      critical_bin_indices, critical_bin_distances_augmented);
  Kokkos::resize(sod_halo_bin_masses, 0, 0); // free as not used afterwards
  Kokkos::resize(particle_masses, 0);        // free as not used afterwards

  Kokkos::Profiling::pushRegion("ArborX::SOD::find_sod_particles");

  // Compute number of particles within SOD's R_delta
  Kokkos::resize(Kokkos::WithoutInitializing, out.sod_particles_offsets,
                 num_halos + 1);
  Kokkos::parallel_for(
      "ArborX::SOD::compute_sod_particles_counts",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_halos),
      KOKKOS_LAMBDA(int halo_index) {
        int count = 0;

        // Sum the counts for all bins smaller than critical one
        for (int bin_id = 0; bin_id < critical_bin_ids(halo_index); ++bin_id)
          count += out.sod_halo_bin_counts(halo_index, bin_id);

        // Add the count for the critical bin
        count += sod_halo_rdeltas_index(halo_index);

        out.sod_particles_offsets(halo_index) = count;
      });
  exclusivePrefixSum(exec_space, out.sod_particles_offsets);

  Kokkos::resize(out.sod_particles_indices,
                 lastElement(out.sod_particles_offsets));

  // Copy critical bin particles
  Kokkos::parallel_for(
      "ArborX::SOD::copy_critical_bin_particles",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_halos),
      KOKKOS_LAMBDA(int halo_index) {
        int num_valid_particles_in_critical_bin =
            sod_halo_rdeltas_index(halo_index);
        int critical_bin_offset = out.sod_particles_offsets(halo_index + 1) -
                                  num_valid_particles_in_critical_bin;

        // TODO: linear scan, may be expensive
        for (int i = 0; i < num_valid_particles_in_critical_bin; ++i)
          out.sod_particles_indices(critical_bin_offset++) =
              critical_bin_indices(critical_bin_offsets(halo_index) + i);
      });
  Kokkos::resize(critical_bin_offsets, 0); // free as not used afterwards
  Kokkos::resize(critical_bin_indices, 0); // free as not used afterwards

  {
    auto offsets = clone(out.sod_particles_offsets);
    bvh.query(
        exec_space, ParticlesWrapper<Particles>{particles},
        SODParticles<MemorySpace, Particles>{
            particles, offsets, out.sod_particles_indices, critical_bin_ids,
            fof_halo_centers, r_min, r_max, num_sod_bins},
        Experimental::TraversalPolicy().setPredicateSorting(sort_predicates));
  }
  Kokkos::Profiling::popRegion();
}

} // namespace SOD

template <typename ExecutionSpace, typename Particles, typename ParticleMasses,
          typename FOFHaloCenters, typename FOFHaloMasses>
auto sod(ExecutionSpace const &exec_space, Particles particles,
         ParticleMasses particle_masses, FOFHaloCenters fof_halo_centers,
         FOFHaloMasses fof_halo_masses, SOD::Parameters const &params)
{
  Kokkos::Profiling::pushRegion("ArborX::SOD");

  using MemorySpace = typename ExecutionSpace::memory_space;

  SODOutputData<MemorySpace> out_device;

  // Transfer data from host to device
  Kokkos::Profiling::pushRegion("ArborX::SOD::copy_input_data_to_device");
  Kokkos::View<ArborX::Point *, MemorySpace> particles_device =
      Kokkos::create_mirror_view_and_copy(exec_space, particles);
  Kokkos::View<float *, MemorySpace> particle_masses_device =
      Kokkos::create_mirror_view_and_copy(exec_space, particle_masses);
  Kokkos::View<ArborX::Point *, MemorySpace> fof_halo_centers_device =
      Kokkos::create_mirror_view_and_copy(exec_space, fof_halo_centers);
  Kokkos::View<float *, MemorySpace> fof_halo_masses_device =
      Kokkos::create_mirror_view_and_copy(exec_space, fof_halo_masses);
  Kokkos::Profiling::popRegion();

  // Execute the kernels on the device
  SOD::sodCore(exec_space, particles_device, particle_masses_device,
               fof_halo_centers_device, fof_halo_masses_device, out_device,
               params);

  SODOutputData<Kokkos::HostSpace> out_host;

  // Transfer data from device to host
  Kokkos::Profiling::pushRegion("ArborX::SOD::copy_output_data_from_device");
  auto copy_bins_to_host = [](auto &view_host, auto &view_device) {
    auto view_mirror =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view_device);
    Kokkos::resize(Kokkos::WithoutInitializing, view_host,
                   view_device.extent(0), view_device.extent(1));
    Kokkos::deep_copy(view_host, view_mirror);
  };
  copy_bins_to_host(out_host.sod_halo_bin_outer_radii,
                    out_device.sod_halo_bin_outer_radii);
  copy_bins_to_host(out_host.sod_halo_bin_masses,
                    out_device.sod_halo_bin_masses);
  copy_bins_to_host(out_host.sod_halo_bin_rhos, out_device.sod_halo_bin_rhos);
  copy_bins_to_host(out_host.sod_halo_bin_rho_ratios,
                    out_device.sod_halo_bin_rho_ratios);
  copy_bins_to_host(out_host.sod_halo_bin_counts,
                    out_device.sod_halo_bin_counts);

  auto copy_to_host = [](auto &view_host, auto &view_device) {
    Kokkos::resize(Kokkos::WithoutInitializing, view_host,
                   view_device.extent(0));
    Kokkos::deep_copy(view_host, view_device);
  };
  copy_to_host(out_host.sod_halo_rdeltas, out_device.sod_halo_rdeltas);
  copy_to_host(out_host.sod_particles_offsets,
               out_device.sod_particles_offsets);
  copy_to_host(out_host.sod_particles_indices,
               out_device.sod_particles_indices);
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::popRegion();

  return out_host;
}

} // namespace ArborX

#endif
