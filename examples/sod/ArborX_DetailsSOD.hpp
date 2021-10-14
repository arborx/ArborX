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

#ifndef ARBORX_DETAILSSOD_HPP
#define ARBORX_DETAILSSOD_HPP

#include <ArborX.hpp>

int constexpr NUM_SOD_BINS = 20 + 1;

namespace ArborX
{

namespace Details
{

KOKKOS_INLINE_FUNCTION
float rDelta(float r_min, float r_max)
{
  return log10(r_max / r_min) / (NUM_SOD_BINS - 1);
}

KOKKOS_INLINE_FUNCTION
int binID(float r_min, float r_max, float r)
{

  float r_delta = rDelta(r_min, r_max);

  int bin_id = 0;
  if (r > r_min)
    bin_id = (int)floor(log10(r / r_min) / r_delta) + 1;
  if (bin_id >= NUM_SOD_BINS)
    bin_id = NUM_SOD_BINS - 1;

  return bin_id;
}

template <typename MemorySpace, typename Particles>
struct BinAccumulator
{
  Particles _particles;
  Kokkos::View<float *, MemorySpace> _particle_masses;
  Kokkos::View<int * [NUM_SOD_BINS], MemorySpace> _sod_halo_bin_counts;
  Kokkos::View<double * [NUM_SOD_BINS], MemorySpace> _sod_halo_bin_masses;
  Kokkos::View<double * [NUM_SOD_BINS], MemorySpace> _sod_halo_bin_avg_radii;
  Kokkos::View<Point *, MemorySpace> _fof_halo_centers;
  float _r_min;
  Kokkos::View<float *, MemorySpace> _r_max;

  using ParticlesAccess = AccessTraits<Particles, PrimitivesTag>;

  template <typename Query>
  KOKKOS_FUNCTION void operator()(Query const &query, int halo_index) const
  {
    auto particle_index = getData(query);
    Point const &point = ParticlesAccess::get(_particles, particle_index);

    float dist = Details::distance(point, _fof_halo_centers(halo_index));
    if (dist > _r_max(halo_index))
    {
      // False positive
      return;
    }

    int bin_id = binID(_r_min, _r_max(halo_index), dist);
    Kokkos::atomic_fetch_add(&_sod_halo_bin_counts(halo_index, bin_id), 1);
    Kokkos::atomic_fetch_add(&_sod_halo_bin_masses(halo_index, bin_id),
                             _particle_masses(particle_index));
    Kokkos::atomic_fetch_add(&_sod_halo_bin_avg_radii(halo_index, bin_id),
                             dist);
  }
};

template <typename MemorySpace, typename Particles>
struct OverlapCount
{
  Particles _particles;
  Kokkos::View<int *, MemorySpace> _counts;
  Kokkos::View<Point *, MemorySpace> _centers;
  Kokkos::View<float *, MemorySpace> _radii;

  using ParticlesAccess = AccessTraits<Particles, PrimitivesTag>;

  template <typename Query>
  KOKKOS_FUNCTION auto operator()(Query const &query, int halo_index) const
  {
    auto particle_index = getData(query);

    Point const &particle = ParticlesAccess::get(_particles, particle_index);
    if (Details::distance(particle, _centers(halo_index)) <= _radii(halo_index))
      ++_counts(particle_index);
  }
};

template <typename MemorySpace, typename Particles>
struct CriticalBinParticles
{
  Particles _particles;
  Kokkos::View<int *, MemorySpace> _offsets;
  Kokkos::View<int *, MemorySpace> _indices;
  Kokkos::View<float *, MemorySpace> _distances_augmented;
  Kokkos::View<int *, MemorySpace> _critical_bin_ids;
  Kokkos::View<Point *, MemorySpace> _fof_halo_centers;
  Kokkos::View<float * [NUM_SOD_BINS], MemorySpace> _sod_halo_bin_outer_radii;
  float _r_min;
  Kokkos::View<float *, MemorySpace> _r_max;

  using ParticlesAccess = AccessTraits<Particles, PrimitivesTag>;

  template <typename Query>
  KOKKOS_FUNCTION auto operator()(Query const &query, int halo_index) const
  {
    auto particle_index = getData(query);

    Point const &particle = ParticlesAccess::get(_particles, particle_index);

    float dist = Details::distance(particle, _fof_halo_centers(halo_index));
    if (dist > _r_max(halo_index))
    {
      // False positive
      return;
    }

    auto bin_id = binID(_r_min, _r_max(halo_index), dist);
    if (bin_id == _critical_bin_ids(halo_index))
    {
      auto pos = Kokkos::atomic_fetch_add(&_offsets(halo_index), 1);
      _indices(pos) = particle_index;

      // NOTE: this is a HACK!!
      // Instead of storing just the distance, we adjust the distance by
      // scaling it to [0, 1) and adding the halo index. We call these
      // distances augmented. This guarantees that the segments with augmented
      // distances for each halo do not overlap, allowing us to then simply use
      // a single sortObjects call that would automatically sort all distances
      // for each halo without mixing them.
      //
      // The reason it is done here, instead of later, is because doing it
      // later would require a linear scan through each halo by a single
      // thread, which is extremely expensive compared to all other kernels.
      //
      // The way we scale each distance to [0, 1) is by noticing that we are
      // only interested in particles in a critical bin. Thus, taking the outer
      // radius of that bin guarantees this property. And just to be sure, we
      // conservatively scale by 1.1.
      _distances_augmented(pos) =
          halo_index +
          dist / (1.1f * _sod_halo_bin_outer_radii(halo_index, bin_id));
    }
  }
};

// Compute R_min and R_max for each FOF halo
template <typename ExecutionSpace, typename FOFHaloMases>
std::pair<float, Kokkos::View<float *, typename FOFHaloMases::memory_space>>
computeSODRadii(ExecutionSpace const &exec_space,
                FOFHaloMases const &fof_halo_masses)
{
  using MemorySpace = typename FOFHaloMases::memory_space;

  // HACC constants
  float constexpr MIN_FACTOR = 0.05;
  float constexpr MAX_FACTOR = 2.0;
  float constexpr R_SMOOTH =
      250.f / 3072; // interparticle separation, rl/np, where rl is the boxsize
                    // of the simulation, and np is the number of particles
  float constexpr SOD_MASS = 1e14;

  float r_min = MIN_FACTOR * R_SMOOTH;

  auto const num_halos = fof_halo_masses.extent(0);
  Kokkos::View<float *, MemorySpace> r_max(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "ArborX::SOD::r_max"),
      num_halos);
  Kokkos::parallel_for(
      "ArborX::SOD::compute_r_max",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_halos),
      KOKKOS_LAMBDA(int i) {
        float R_init = std::cbrt(fof_halo_masses(i) / SOD_MASS);
        r_max(i) = MAX_FACTOR * R_init;
      });

  return std::make_pair(r_min, r_max);
}

// Compute rho and rho_ratio
template <typename ExecutionSpace, typename MemorySpace>
std::tuple<Kokkos::View<float * [NUM_SOD_BINS], MemorySpace>,
           Kokkos::View<float *[NUM_SOD_BINS], MemorySpace>>
computeSODRhos(
    ExecutionSpace const &exec_space, float RHO,
    Kokkos::View<double * [NUM_SOD_BINS], MemorySpace> sod_halo_bin_masses,
    Kokkos::View<double * [NUM_SOD_BINS], MemorySpace> sod_halo_bin_avg_radii) {
  auto const num_halos = sod_halo_bin_masses.extent(0);

  Kokkos::View<float * [NUM_SOD_BINS], MemorySpace> sod_halo_bin_rhos(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "ArborX::SOD::sod_halo_bin_rhos"),
      num_halos);
  Kokkos::View<float * [NUM_SOD_BINS], MemorySpace> sod_halo_bin_rho_ratios(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "ArborX::SOD::sod_halo_bin_rho_ratios"),
      num_halos);
  Kokkos::parallel_for(
      "ArborX::SOD::compute_rhos",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_halos),
      KOKKOS_LAMBDA(int halo_index) {
        double accumulated_mass = 0.;
        for (int bin_id = 0; bin_id < NUM_SOD_BINS; ++bin_id)
        {
          auto &rho = sod_halo_bin_rhos(halo_index, bin_id);
          auto &rho_ratio = sod_halo_bin_rho_ratios(halo_index, bin_id);

          accumulated_mass += sod_halo_bin_masses(halo_index, bin_id);
          auto avg_radius = sod_halo_bin_avg_radii(halo_index, bin_id);
          auto volume = 4.f / 3 * M_PI * pow(avg_radius, 3);

          sod_halo_bin_rhos(halo_index, bin_id) =
              ((accumulated_mass > 0 && volume > 0) ? accumulated_mass / volume
                                                    : 0);
          rho_ratio = rho / RHO;
        }
      });

  return std::make_pair(sod_halo_bin_rhos, sod_halo_bin_rho_ratios);
}

// Compute critical bins
template <typename ExecutionSpace, typename MemorySpace>
Kokkos::View<int *, MemorySpace> computeSODCriticalBins(
    ExecutionSpace const &exec_space, float RHO, float DELTA,
    Kokkos::View<double * [NUM_SOD_BINS], MemorySpace> sod_halo_bin_masses,
    Kokkos::View<int * [NUM_SOD_BINS], MemorySpace> sod_halo_bin_counts,
    Kokkos::View<float * [NUM_SOD_BINS], MemorySpace> sod_halo_bin_outer_radii)
{
  auto const num_halos = sod_halo_bin_masses.extent(0);

  Kokkos::View<int *, MemorySpace> critical_bin_ids(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "ArborX::SOD::critical_bin_ids"),
      num_halos);
  Kokkos::deep_copy(critical_bin_ids, -1);
  Kokkos::parallel_for(
      "ArborX::SOD::compute_critical_bins",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_halos),
      KOKKOS_LAMBDA(int halo_index) {
        int &critical_bin_id = critical_bin_ids(halo_index);

        double accumulated_mass = 0.;
        for (int bin_id = 0; bin_id < NUM_SOD_BINS; ++bin_id)
        {
          accumulated_mass += sod_halo_bin_masses(halo_index, bin_id);
          float outer_radius = sod_halo_bin_outer_radii(halo_index, bin_id);

          float bin_rho_ratio_int = 0;
          if (outer_radius > 0)
          {
            auto volume = 4.f / 3 * M_PI * pow(outer_radius, 3);
            bin_rho_ratio_int = (accumulated_mass / volume) / RHO;
          }

          if (bin_rho_ratio_int <= DELTA)
          {
            critical_bin_id = bin_id;
            break;
          }
        }

        if (critical_bin_id < 0)
        {
          printf("%d (halo tag ?): max radius is not big enough, will "
                 "underestimate\n",
                 halo_index);
          critical_bin_id = NUM_SOD_BINS - 1;
          while (sod_halo_bin_counts(halo_index, critical_bin_id) == 0)
            --critical_bin_id;
        }
        else if (critical_bin_id == 0)
        {
          printf("%d (halo tag ?): min radius is not small enough, will "
                 "overestimate\n",
                 halo_index);
          while (sod_halo_bin_counts(halo_index, critical_bin_id) == 0)
            ++critical_bin_id;
        }

        // Capture the cases where normal critical bin is empty, and that
        // extra radius meets the rho_ratio criteria
        while (sod_halo_bin_counts(halo_index, critical_bin_id) == 0)
          --critical_bin_id;

        // This should not happen but protecting against no zero bins for all up
        // to the critical bin
        if (critical_bin_id < 0)
          printf(
              "%d (halo tag ?): error, not even the first bin has particles\n",
              halo_index);
      });

  return critical_bin_ids;
}

// Compute radii for SOD bins
template <typename ExecutionSpace, typename MemorySpace>
Kokkos::View<float *[NUM_SOD_BINS], MemorySpace>
computeSODBinRadii(ExecutionSpace const &exec_space, float r_min,
                   Kokkos::View<float *, MemorySpace> r_max) {
  auto const num_halos = r_max.extent(0);

  Kokkos::View<float * [NUM_SOD_BINS], MemorySpace> sod_halo_bin_outer_radii(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "ArborX::SOD::sod_halo_bin_outer_radii"),
      num_halos);
  Kokkos::parallel_for(
      "ArborX::SOD::compute_bin_outer_radii",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_halos),
      KOKKOS_LAMBDA(int halo_index) {
        float r_delta = rDelta(r_min, r_max(halo_index));
        sod_halo_bin_outer_radii(halo_index, 0) = r_min;
        for (int bin_id = 1; bin_id < NUM_SOD_BINS; ++bin_id)
          sod_halo_bin_outer_radii(halo_index, bin_id) =
              pow(10.0, bin_id * r_delta) * r_min;
      });

  return sod_halo_bin_outer_radii;
}

template <typename ExecutionSpace, typename MemorySpace>
Kokkos::View<float *, MemorySpace> computeSODRdeltas(
    ExecutionSpace const &exec_space, float DELTA, float RHO,
    Kokkos::View<float *, MemorySpace> particle_masses,
    Kokkos::View<double * [NUM_SOD_BINS], MemorySpace> sod_halo_bin_masses,
    Kokkos::View<float * [NUM_SOD_BINS], MemorySpace> sod_halo_bin_outer_radii,
    Kokkos::View<int *, MemorySpace> critical_bin_ids,
    Kokkos::View<int *, MemorySpace> critical_bin_offsets,
    Kokkos::View<int *, MemorySpace> critical_bin_indices,
    Kokkos::View<float *, MemorySpace> critical_bin_distances_augmented)
{
  using HostExecutionSpace = Kokkos::DefaultHostExecutionSpace;

  HostExecutionSpace host_space;

  auto num_halos = critical_bin_offsets.extent(0) - 1;
  auto num_critical_bin_particles = critical_bin_indices.extent(0);

  // Permute found particles within a critical bin of each halo based on their
  // distance to the center
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

  // Migrate the data to the host
  auto critical_bin_ids_host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, critical_bin_ids);
  auto critical_bin_offsets_host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, critical_bin_offsets);
  auto critical_bin_indices_host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, critical_bin_indices);
  auto critical_bin_distances_augmented_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{},
                                          critical_bin_distances_augmented);
  auto sod_halo_bin_outer_radii_host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, sod_halo_bin_outer_radii);
  auto sod_halo_bin_masses_host =
      Kokkos::create_mirror_view_and_copy(host_space, sod_halo_bin_masses);
  auto particle_masses_host =
      Kokkos::create_mirror_view_and_copy(host_space, particle_masses);

  // Compute R_delta
  Kokkos::View<float *, Kokkos::HostSpace> sod_halo_rdeltas_host(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "ArborX::SOD::r_delta"),
      num_halos);
  Kokkos::parallel_for(
      "ArborX::SOD::compute_r_delta",
      Kokkos::RangePolicy<HostExecutionSpace>(host_space, 0, num_halos),
      KOKKOS_LAMBDA(int halo_index) {
        double mass = 0.f;
        for (int bin_id = 0; bin_id < critical_bin_ids_host(halo_index);
             ++bin_id)
          mass += sod_halo_bin_masses_host(halo_index, bin_id);

        float R_bin_outer = sod_halo_bin_outer_radii_host(
            halo_index, critical_bin_ids_host(halo_index));

        auto bin_start = critical_bin_offsets_host(halo_index);
        auto bin_end = critical_bin_offsets_host(halo_index + 1);
        assert(bin_start < bin_end);

        // By default, set the r_delta to be the last particle in the bin. This
        // fixes a potential error of r_200 between the bin edge and the first
        // particle radius.
        sod_halo_rdeltas_host(halo_index) =
            (critical_bin_distances_augmented_host(bin_end - 1) - halo_index) *
            (1.1f * R_bin_outer); // see HACK comment above for details

        for (int i = bin_start; i < bin_end; ++i)
        {
          mass += particle_masses_host(critical_bin_indices_host(i));
          float r = (critical_bin_distances_augmented_host(i) - halo_index) *
                    (1.1f * R_bin_outer); // see HACK comment above for details

          float volume = 4.f / 3 * M_PI * pow(r, 3);
          float ratio = (mass / volume) / RHO;

          if (ratio <= DELTA)
          {
            sod_halo_rdeltas_host(halo_index) = r;
            break;
          }
        }
      });

  Kokkos::View<float *, MemorySpace> sod_halo_rdeltas_device(
      "ArborX::SOD::r_delta", num_halos);
  Kokkos::deep_copy(sod_halo_rdeltas_device, sod_halo_rdeltas_host);

  return sod_halo_rdeltas_device;
}

} // namespace Details

} // namespace ArborX

#endif
