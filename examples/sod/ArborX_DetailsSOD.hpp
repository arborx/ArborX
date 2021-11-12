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

namespace ArborX
{

namespace SOD
{

struct Parameters
{
  // Number of bins to profile
  int _num_sod_bins = 20 + 1;
  // Interparticle separation. Equal to rl/np, where rl is the boxsize of the
  // simulation, and np is the number of particles
  float _r_smooth = -1.f;
  float _rho = -1.f;
  float _sod_mass = -1.f;
  float _rho_ratio = 200;
  float _min_factor = 0.05;
  float _max_factor = 2.0;

  Parameters &setNumSODBins(int num_sod_bins)
  {
    ARBORX_ASSERT(num_sod_bins > 0);
    _num_sod_bins = num_sod_bins;
    return *this;
  }
  Parameters &setRSmooth(float r_smooth)
  {
    ARBORX_ASSERT(r_smooth > 0);
    _r_smooth = r_smooth;
    return *this;
  }
  Parameters &setRho(float rho)
  {
    ARBORX_ASSERT(rho > 0);
    _rho = rho;
    return *this;
  }
  Parameters &setSODMass(float sod_mass)
  {
    ARBORX_ASSERT(sod_mass > 0);
    _sod_mass = sod_mass;
    return *this;
  }
  Parameters &setRhoRatio(float rho_ratio)
  {
    ARBORX_ASSERT(rho_ratio > 0);
    _rho_ratio = rho_ratio;
    return *this;
  }
  Parameters &setMinFactor(float min_factor)
  {
    ARBORX_ASSERT(min_factor > 0);
    _min_factor = min_factor;
    return *this;
  }
  Parameters &setMaxFactor(float max_factor)
  {
    ARBORX_ASSERT(max_factor > 0);
    _max_factor = max_factor;
    return *this;
  }
};

KOKKOS_INLINE_FUNCTION
float rDelta(float r_min, float r_max, int num_sod_bins)
{
  return log10(r_max / r_min) / (num_sod_bins - 1);
}

KOKKOS_INLINE_FUNCTION
int binID(float r_min, float r_max, float r, int num_sod_bins)
{

  float r_delta = rDelta(r_min, r_max, num_sod_bins);

  int bin_id = 0;
  if (r > r_min)
    bin_id = (int)floor(log10(r / r_min) / r_delta) + 1;
  if (bin_id >= num_sod_bins)
    bin_id = num_sod_bins - 1;

  return bin_id;
}

template <typename MemorySpace, typename Particles>
struct BinAccumulator
{
  Particles _particles;
  Kokkos::View<float *, MemorySpace> _particle_masses;
  Kokkos::View<int **, MemorySpace> _sod_halo_bin_counts;
  Kokkos::View<double **, MemorySpace> _sod_halo_bin_masses;
  Kokkos::View<double **, MemorySpace> _sod_halo_bin_avg_radii;
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

    int const num_sod_bins = _sod_halo_bin_masses.extent(1);

    int bin_id = binID(_r_min, _r_max(halo_index), dist, num_sod_bins);
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
  Kokkos::View<int *, MemorySpace> _critical_bin_offsets;
  Kokkos::View<int *, MemorySpace> _critical_bin_indices;
  Kokkos::View<float *, MemorySpace> _distances_augmented;
  Kokkos::View<int *, MemorySpace> _critical_bin_ids;
  Kokkos::View<Point *, MemorySpace> _fof_halo_centers;
  Kokkos::View<float **, MemorySpace> _sod_halo_bin_outer_radii;
  float _r_min;
  Kokkos::View<float *, MemorySpace> _r_max;

  using ParticlesAccess = AccessTraits<Particles, PrimitivesTag>;

  template <typename Query>
  KOKKOS_FUNCTION void operator()(Query const &query, int halo_index) const
  {
    auto particle_index = getData(query);

    Point const &particle = ParticlesAccess::get(_particles, particle_index);

    float dist = Details::distance(particle, _fof_halo_centers(halo_index));
    if (dist > _r_max(halo_index))
    {
      // False positive
      return;
    }

    int const num_sod_bins = _sod_halo_bin_outer_radii.extent(1);

    auto bin_id = binID(_r_min, _r_max(halo_index), dist, num_sod_bins);
    if (bin_id == _critical_bin_ids(halo_index))
    {
      auto pos =
          Kokkos::atomic_fetch_add(&_critical_bin_offsets(halo_index), 1);
      _critical_bin_indices(pos) = particle_index;

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

template <typename MemorySpace, typename Particles>
struct SODParticles
{
  Particles _particles;
  Kokkos::View<int *, MemorySpace> _sod_particles_offsets;
  Kokkos::View<int *, MemorySpace> _sod_particles_indices;
  Kokkos::View<int *, MemorySpace> _critical_bin_ids;
  Kokkos::View<Point *, MemorySpace> _fof_halo_centers;
  float _r_min;
  Kokkos::View<float *, MemorySpace> _r_max;
  int _num_sod_bins;

  using ParticlesAccess = AccessTraits<Particles, PrimitivesTag>;

  template <typename Query>
  KOKKOS_FUNCTION void operator()(Query const &query, int halo_index) const
  {
    auto particle_index = getData(query);

    Point const &particle = ParticlesAccess::get(_particles, particle_index);

    float dist = Details::distance(particle, _fof_halo_centers(halo_index));
    if (dist > _r_max(halo_index))
    {
      // False positive
      return;
    }

    auto bin_id = binID(_r_min, _r_max(halo_index), dist, _num_sod_bins);
    if (bin_id < _critical_bin_ids(halo_index))
    {
      auto pos =
          Kokkos::atomic_fetch_add(&_sod_particles_offsets(halo_index), 1);
      _sod_particles_indices(pos) = particle_index;
    }
  }
};

// Compute R_min and R_max for each FOF halo
template <typename ExecutionSpace, typename FOFHaloMases>
std::pair<float, Kokkos::View<float *, typename FOFHaloMases::memory_space>>
computeSODRadii(ExecutionSpace const &exec_space, Parameters const &params,
                FOFHaloMases const &fof_halo_masses)
{
  Kokkos::Profiling::pushRegion("ArborX::SOD::compute_sod_radii");

  using MemorySpace = typename FOFHaloMases::memory_space;

  float r_min = params._min_factor * params._r_smooth;

  auto const num_halos = fof_halo_masses.extent(0);
  Kokkos::View<float *, MemorySpace> r_max(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "ArborX::SOD::r_max"),
      num_halos);
  Kokkos::parallel_for(
      "ArborX::SOD::compute_r_max",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_halos),
      KOKKOS_LAMBDA(int i) {
        float R_init = std::cbrt(fof_halo_masses(i) / params._sod_mass);
        r_max(i) = params._max_factor * R_init;
      });

  Kokkos::Profiling::popRegion();

  return std::make_pair(r_min, r_max);
}

// Compute critical bins
template <typename ExecutionSpace, typename MemorySpace>
Kokkos::View<int *, MemorySpace> computeSODCriticalBins(
    ExecutionSpace const &exec_space, Parameters const &params,
    Kokkos::View<double **, MemorySpace> sod_halo_bin_masses,
    Kokkos::View<int **, MemorySpace> sod_halo_bin_counts,
    Kokkos::View<float **, MemorySpace> sod_halo_bin_outer_radii)
{
  Kokkos::Profiling::pushRegion("ArborX::SOD::compute_critical_bins");

  auto const num_halos = sod_halo_bin_masses.extent(0);
  int const num_sod_bins = sod_halo_bin_masses.extent(1);

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
        for (int bin_id = 0; bin_id < num_sod_bins; ++bin_id)
        {
          accumulated_mass += sod_halo_bin_masses(halo_index, bin_id);
          float outer_radius = sod_halo_bin_outer_radii(halo_index, bin_id);

          float bin_rho_ratio_int = 0;
          if (outer_radius > 0)
          {
            auto volume = 4.f / 3 * M_PI * pow(outer_radius, 3);
            bin_rho_ratio_int = (accumulated_mass / volume) / params._rho;
          }

          if (bin_rho_ratio_int <= params._rho_ratio)
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
          critical_bin_id = num_sod_bins - 1;
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

  Kokkos::Profiling::popRegion();

  return critical_bin_ids;
}

// Compute radii for SOD bins
template <typename ExecutionSpace, typename MemorySpace>
Kokkos::View<float **, MemorySpace>
computeSODBinRadii(ExecutionSpace const &exec_space, float r_min,
                   Kokkos::View<float *, MemorySpace> r_max, int num_sod_bins)
{
  Kokkos::Profiling::pushRegion("ArborX::SOD::compute_sod_bin_radii");

  auto const num_halos = r_max.extent(0);

  Kokkos::View<float **, MemorySpace> sod_halo_bin_outer_radii(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "ArborX::SOD::sod_halo_bin_outer_radii"),
      num_halos, num_sod_bins);
  Kokkos::parallel_for(
      "ArborX::SOD::compute_bin_outer_radii",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_halos),
      KOKKOS_LAMBDA(int halo_index) {
        float r_delta = rDelta(r_min, r_max(halo_index), num_sod_bins);
        sod_halo_bin_outer_radii(halo_index, 0) = r_min;
        for (int bin_id = 1; bin_id < num_sod_bins; ++bin_id)
          sod_halo_bin_outer_radii(halo_index, bin_id) =
              pow(10.0, bin_id * r_delta) * r_min;
      });

  Kokkos::Profiling::popRegion();

  return sod_halo_bin_outer_radii;
}

template <typename ExecutionSpace, typename MemorySpace>
std::pair<Kokkos::View<float *, MemorySpace>, Kokkos::View<int *, MemorySpace>>
computeSODRdeltas(
    ExecutionSpace const & /*exec_space*/, Parameters const &params,
    Kokkos::View<float *, MemorySpace> particle_masses,
    Kokkos::View<double **, MemorySpace> sod_halo_bin_masses,
    Kokkos::View<float **, MemorySpace> sod_halo_bin_outer_radii,
    Kokkos::View<int *, MemorySpace> critical_bin_ids,
    Kokkos::View<int *, MemorySpace> critical_bin_offsets,
    Kokkos::View<int *, MemorySpace> sorted_critical_bin_indices,
    Kokkos::View<float *, MemorySpace> critical_bin_distances_augmented)
{
  Kokkos::Profiling::pushRegion("ArborX::SOD::compute_r_delta");

  using HostExecutionSpace = Kokkos::DefaultHostExecutionSpace;

  HostExecutionSpace host_space;

  auto num_halos = critical_bin_offsets.extent(0) - 1;

  // Migrate the data to the host
  auto critical_bin_ids_host =
      Kokkos::create_mirror_view_and_copy(host_space, critical_bin_ids);
  auto critical_bin_offsets_host =
      Kokkos::create_mirror_view_and_copy(host_space, critical_bin_offsets);
  auto critical_bin_indices_host = Kokkos::create_mirror_view_and_copy(
      host_space, sorted_critical_bin_indices);
  auto critical_bin_distances_augmented_host =
      Kokkos::create_mirror_view_and_copy(host_space,
                                          critical_bin_distances_augmented);
  auto sod_halo_bin_outer_radii_host =
      Kokkos::create_mirror_view_and_copy(host_space, sod_halo_bin_outer_radii);
  auto sod_halo_bin_masses_host =
      Kokkos::create_mirror_view_and_copy(host_space, sod_halo_bin_masses);
  auto particle_masses_host =
      Kokkos::create_mirror_view_and_copy(host_space, particle_masses);

  // Compute R_delta
  Kokkos::View<float *, Kokkos::HostSpace> sod_halo_rdeltas_host(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "ArborX::SOD::r_delta"),
      num_halos);
  Kokkos::View<int *, Kokkos::HostSpace> sod_halo_rdeltas_index_host(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "ArborX::SOD::r_delta_index"),
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
        sod_halo_rdeltas_index_host(halo_index) = (bin_end - 1 - bin_start);

        for (int i = bin_start; i < bin_end; ++i)
        {
          mass += particle_masses_host(critical_bin_indices_host(i));
          float r = (critical_bin_distances_augmented_host(i) - halo_index) *
                    (1.1f * R_bin_outer); // see HACK comment above for details

          float volume = 4.f / 3 * M_PI * pow(r, 3);
          float ratio = (mass / volume) / params._rho;

          if (ratio <= params._rho_ratio)
          {
            sod_halo_rdeltas_host(halo_index) = r;
            sod_halo_rdeltas_index_host(halo_index) = (i - bin_start);
            break;
          }
        }
      });

  Kokkos::View<float *, MemorySpace> sod_halo_rdeltas_device(
      "ArborX::SOD::r_delta", num_halos);
  Kokkos::deep_copy(sod_halo_rdeltas_device, sod_halo_rdeltas_host);

  Kokkos::View<int *, MemorySpace> sod_halo_rdeltas_index_device(
      "ArborX::SOD::r_delta_index", num_halos);
  Kokkos::deep_copy(sod_halo_rdeltas_index_device, sod_halo_rdeltas_index_host);

  Kokkos::Profiling::popRegion();

  return std::make_pair(sod_halo_rdeltas_device, sod_halo_rdeltas_index_device);
}

} // namespace SOD

} // namespace ArborX

#endif
