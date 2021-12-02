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
  float _rho = -1.f;
  float _rho_ratio = 200;

  Parameters &setRho(float rho)
  {
    ARBORX_ASSERT(rho > 0);
    _rho = rho;
    return *this;
  }
  Parameters &setRhoRatio(float rho_ratio)
  {
    ARBORX_ASSERT(rho_ratio > 0);
    _rho_ratio = rho_ratio;
    return *this;
  }
};

} // namespace SOD

namespace Details
{

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

// Compute critical bins
template <typename ExecutionSpace, typename MemorySpace>
Kokkos::View<int *, MemorySpace> computeSODCriticalBins(
    ExecutionSpace const &exec_space, SOD::Parameters const &params,
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
          while (sod_halo_bin_counts(halo_index, critical_bin_id) == 0 &&
                 critical_bin_id > 0)
            --critical_bin_id;
        }
        else if (critical_bin_id == 0)
        {
          printf("%d (halo tag ?): min radius is not small enough, will "
                 "overestimate\n",
                 halo_index);
          while (sod_halo_bin_counts(halo_index, critical_bin_id) == 0 &&
                 critical_bin_id < num_sod_bins - 1)
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
    ExecutionSpace const & /*exec_space*/, SOD::Parameters const &params,
    Kokkos::View<float *, MemorySpace> particle_masses,
    Kokkos::View<double **, MemorySpace> sod_halo_bin_masses,
    Kokkos::View<float **, MemorySpace> sod_halo_bin_outer_radii,
    Kokkos::View<int *, MemorySpace> critical_bin_ids,
    Kokkos::View<int *, MemorySpace> critical_bin_offsets,
    Kokkos::View<int *, MemorySpace> sorted_critical_bin_indices,
    Kokkos::View<float *, MemorySpace> critical_bin_distances_augmented)
{
  Kokkos::Profiling::pushRegion("ArborX::SOD::compute_r_delta");

  using TeamPolicy =
      Kokkos::TeamPolicy<ExecutionSpace, Kokkos::Schedule<Kokkos::Dynamic>>;
  using team_member = typename TeamPolicy::member_type;

  auto num_halos = critical_bin_offsets.extent(0) - 1;

  Kokkos::View<float *, MemorySpace> sod_halo_rdeltas(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "ArborX::SOD::r_delta"),
      num_halos);
  Kokkos::View<int *, MemorySpace> sod_halo_rdeltas_index(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "ArborX::SOD::r_delta_index"),
      num_halos);
  Kokkos::parallel_for(
      "ArborX::SODHandle::computeRdelta::compute_rdelta_index",
      TeamPolicy(num_halos, Kokkos::AUTO),
      KOKKOS_LAMBDA(const team_member &team) {
        auto halo_index = team.league_rank();

        auto bin_start = critical_bin_offsets(halo_index);
        auto bin_end = critical_bin_offsets(halo_index + 1);

        // Assert that the critical bin is not empty
        assert(bin_end - bin_start > 0);

        // By default, set the r_delta to be the last particle in the bin.
        // This fixes a potential error of r_200 between the bin edge and the
        // first particle radius.
        Kokkos::single(Kokkos::PerTeam(team), [&]() {
          sod_halo_rdeltas_index(halo_index) = (bin_end - 1 - bin_start);
        });

        float R_bin_outer =
            sod_halo_bin_outer_radii(halo_index, critical_bin_ids(halo_index));

        double prior_mass = 0.;
        for (int bin_id = 0; bin_id < critical_bin_ids(halo_index); ++bin_id)
          prior_mass += sod_halo_bin_masses(halo_index, bin_id);

        Kokkos::parallel_scan(
            Kokkos::TeamThreadRange(team, bin_start, bin_end),
            [&](int i, double &accumulated_mass, bool const final_pass) {
              auto particle_index = sorted_critical_bin_indices(i);

              accumulated_mass += particle_masses(particle_index);
              if (final_pass)
              {
                float r =
                    (critical_bin_distances_augmented(i) - halo_index) *
                    (1.1f * R_bin_outer); // see HACK comment above for details
                float volume = 4.f / 3 * M_PI * pow(r, 3);
                float ratio =
                    ((prior_mass + accumulated_mass) / params._rho) / volume;

                if (ratio <= params._rho_ratio)
                  Kokkos::atomic_min_fetch(&sod_halo_rdeltas_index(halo_index),
                                           i - bin_start);
              }
            });

        Kokkos::single(Kokkos::PerTeam(team), [&]() {
          auto particle_index = sorted_critical_bin_indices(
              bin_start + sod_halo_rdeltas_index(halo_index));

          sod_halo_rdeltas(halo_index) =
              (critical_bin_distances_augmented(
                   bin_start + sod_halo_rdeltas_index(halo_index)) -
               halo_index) *
              (1.1f * R_bin_outer); // see HACK comment above for details
        });
      });

  Kokkos::Profiling::popRegion();

  return std::make_pair(sod_halo_rdeltas, sod_halo_rdeltas_index);
}

} // namespace Details

} // namespace ArborX

#endif
