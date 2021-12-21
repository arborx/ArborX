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

#ifndef ARBORX_DETAILS_SPHERICAL_OVERDENSITY_HPP
#define ARBORX_DETAILS_SPHERICAL_OVERDENSITY_HPP

#include <ArborX.hpp>

namespace ArborX
{

namespace Details
{

KOKKOS_INLINE_FUNCTION
float rDelta(float r_min, float r_max, int num_bins)
{
  using Kokkos::Experimental::log10;

  return log10(r_max / r_min) / (num_bins - 1);
}

KOKKOS_INLINE_FUNCTION
int binID(float r_min, float r_max, int num_bins, float r)
{
  using Kokkos::Experimental::log10;

  float r_delta = rDelta(r_min, r_max, num_bins);

  int bin_id = 0;
  if (r > r_min)
    bin_id = (int)floor(log10(r / r_min) / r_delta) + 1;
  if (bin_id >= num_bins)
    bin_id = num_bins - 1;

  return bin_id;
}

struct SOTuple
{
  int particle_index;
  int halo_index;
  float distance;

private:
  friend KOKKOS_FUNCTION bool operator<(SOTuple const &l, SOTuple const &r)
  {
    if (l.halo_index == r.halo_index)
      return l.distance < r.distance;
    return l.halo_index < r.halo_index;
  }
};

template <typename MemorySpace, typename Callback>
struct Profiles
{
  Kokkos::View<Point *, MemorySpace> _fof_halo_centers;
  float _r_min;
  Kokkos::View<float *, MemorySpace> _r_max;
  int _num_bins;
  Callback _callback;

  template <typename Query>
  KOKKOS_FUNCTION void operator()(Query const &query, int halo_index) const
  {
    int particle_index = getData(query);
    Point const &point = getGeometry(query);

    float dist = distance(point, _fof_halo_centers(halo_index));
    if (dist > _r_max(halo_index))
    {
      // false positive
      return;
    }

    auto bin_id = binID(_r_min, _r_max(halo_index), _num_bins, dist);
    _callback(particle_index, halo_index, bin_id);
  }
};

template <typename MemorySpace>
struct MassAvgRadiiCountProfiles
{
  Kokkos::View<float *, MemorySpace> _particle_masses;
  Kokkos::View<ArborX::Point *, MemorySpace> _fof_halo_centers;
  Kokkos::View<double **, MemorySpace> _sod_halo_bin_masses;
  Kokkos::View<int **, MemorySpace> _sod_halo_bin_counts;

  KOKKOS_FUNCTION void operator()(int particle_index, int halo_index,
                                  int bin_id) const
  {
    Kokkos::atomic_increment(&_sod_halo_bin_counts(halo_index, bin_id));
    Kokkos::atomic_add(&_sod_halo_bin_masses(halo_index, bin_id),
                       (double)_particle_masses(particle_index));
  }
};

template <typename MemorySpace>
struct SOParticlesCount
{
  Kokkos::View<ArborX::Point *, MemorySpace> _fof_halo_centers;
  Kokkos::View<float *, MemorySpace> _r_max;

  Kokkos::View<int *, MemorySpace> _counts;

  template <typename Query>
  KOKKOS_FUNCTION void operator()(Query const &query, int halo_index) const
  {
    float dist = distance(getGeometry(query), _fof_halo_centers(halo_index));
    if (dist > _r_max(halo_index))
    {
      // False positive
      return;
    }

    Kokkos::atomic_increment(&_counts(halo_index));
  }
};

template <typename MemorySpace>
struct SOParticles
{
  Kokkos::View<ArborX::Point *, MemorySpace> _fof_halo_centers;
  Kokkos::View<float *, MemorySpace> _r_max;

  Kokkos::View<int *, MemorySpace> _offsets;
  Kokkos::View<SOTuple *, MemorySpace> _values;

  template <typename Query>
  KOKKOS_FUNCTION void operator()(Query const &query, int halo_index) const
  {
    float dist = distance(getGeometry(query), _fof_halo_centers(halo_index));
    if (dist > _r_max(halo_index))
    {
      // False positive
      return;
    }

    auto pos = Kokkos::atomic_fetch_add(&_offsets(halo_index), 1);

    int particle_index = getData(query);
    _values(pos).particle_index = particle_index;
    _values(pos).halo_index = halo_index;
    _values(pos).distance = dist;
  }
};

template <typename MemorySpace, typename Particles>
struct CriticalBinParticles
{
  Particles _particles;
  Kokkos::View<int *, MemorySpace> _critical_bin_ids;
  Kokkos::View<Point *, MemorySpace> _fof_halo_centers;
  int _num_sod_bins;
  float _r_min;
  Kokkos::View<float *, MemorySpace> _r_max;

  Kokkos::View<int *, MemorySpace> _critical_bin_offsets;
  Kokkos::View<SOTuple *, MemorySpace> _critical_bin_values;

  using ParticlesAccess = AccessTraits<Particles, PrimitivesTag>;

  template <typename Query>
  KOKKOS_FUNCTION void operator()(Query const &query, int halo_index) const
  {
    float dist = distance(getGeometry(query), _fof_halo_centers(halo_index));
    if (dist > _r_max(halo_index))
    {
      // False positive
      return;
    }

    auto bin_id = binID(_r_min, _r_max(halo_index), _num_sod_bins, dist);
    if (bin_id == _critical_bin_ids(halo_index))
    {
      auto pos =
          Kokkos::atomic_fetch_add(&_critical_bin_offsets(halo_index), 1);
      _critical_bin_values(pos).particle_index = getData(query);
      _critical_bin_values(pos).halo_index = halo_index;
      _critical_bin_values(pos).distance = dist;
    }
  }
};

// Compute radii for SO bins
template <typename ExecutionSpace, typename MemorySpace>
Kokkos::View<float **, MemorySpace>
computeSOBinRadii(ExecutionSpace const &exec_space, float r_min,
                  Kokkos::View<float *, MemorySpace> r_max, int num_sod_bins)
{
  Kokkos::Profiling::pushRegion("ArborX::SO::compute_sod_bin_radii");

  auto const num_halos = r_max.extent(0);

  Kokkos::View<float **, MemorySpace> sod_halo_bin_outer_radii(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "ArborX::SO::sod_halo_bin_outer_radii"),
      num_halos, num_sod_bins);
  Kokkos::parallel_for(
      "ArborX::SO::compute_bin_outer_radii",
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

// Compute critical bins
template <typename ExecutionSpace, typename MemorySpace>
Kokkos::View<int *, MemorySpace> computeSOCriticalBins(
    ExecutionSpace const &exec_space, float rho, float rho_ratio,
    Kokkos::View<double **, MemorySpace> sod_halo_bin_masses,
    Kokkos::View<int **, MemorySpace> sod_halo_bin_counts,
    Kokkos::View<float **, MemorySpace> sod_halo_bin_outer_radii)
{
  Kokkos::Profiling::pushRegion("ArborX::SO::compute_critical_bins");

  auto const num_halos = sod_halo_bin_masses.extent(0);
  int const num_sod_bins = sod_halo_bin_masses.extent(1);

  Kokkos::View<int *, MemorySpace> critical_bin_ids(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "ArborX::SO::critical_bin_ids"),
      num_halos);
  Kokkos::deep_copy(critical_bin_ids, -1);
  Kokkos::parallel_for(
      "ArborX::SO::compute_critical_bins",
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
            bin_rho_ratio_int = (accumulated_mass / volume) / rho;
          }

          if (bin_rho_ratio_int <= rho_ratio)
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

template <typename ExecutionSpace, typename MemorySpace>
std::pair<Kokkos::View<float *, MemorySpace>, Kokkos::View<int *, MemorySpace>>
computeSORdeltas(
    ExecutionSpace const &exec_space, float rho, float rho_ratio,
    Kokkos::View<float *, MemorySpace> particle_masses,
    Kokkos::View<double **, MemorySpace> sod_halo_bin_masses,
    Kokkos::View<int *, MemorySpace> critical_bin_ids,
    Kokkos::View<int *, MemorySpace> critical_bin_offsets,
    Kokkos::View<SOTuple *, MemorySpace> sorted_critical_bin_values)
{
  Kokkos::Profiling::pushRegion("ArborX::SO::compute_r_delta");

  using TeamPolicy =
      Kokkos::TeamPolicy<ExecutionSpace, Kokkos::Schedule<Kokkos::Dynamic>>;
  using team_member = typename TeamPolicy::member_type;

  auto num_halos = critical_bin_offsets.extent(0) - 1;

  Kokkos::View<float *, MemorySpace> sod_halo_rdeltas(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "ArborX::SO::r_delta"),
      num_halos);
  Kokkos::View<int *, MemorySpace> sod_halo_rdeltas_index(
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "ArborX::SO::r_delta_index"),
      num_halos);
  Kokkos::parallel_for(
      "ArborX::SOHandle::computeRdelta::compute_rdelta_index",
      TeamPolicy(exec_space, num_halos, Kokkos::AUTO),
      KOKKOS_LAMBDA(team_member const &team) {
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

        double prior_mass = 0.;
        for (int bin_id = 0; bin_id < critical_bin_ids(halo_index); ++bin_id)
          prior_mass += sod_halo_bin_masses(halo_index, bin_id);

        Kokkos::parallel_scan(
            Kokkos::TeamThreadRange(team, bin_start, bin_end),
            [&](int i, double &accumulated_mass, bool const final_pass) {
              auto particle_index =
                  sorted_critical_bin_values(i).particle_index;

              accumulated_mass += particle_masses(particle_index);
              if (final_pass)
              {
                float r = sorted_critical_bin_values(i).distance;
                float volume = 4.f / 3 * M_PI * pow(r, 3);
                float ratio = ((prior_mass + accumulated_mass) / rho) / volume;

                if (ratio <= rho_ratio)
                  Kokkos::atomic_min(&sod_halo_rdeltas_index(halo_index),
                                     i - bin_start);
              }
            });

        Kokkos::single(Kokkos::PerTeam(team), [&]() {
          auto index = bin_start + sod_halo_rdeltas_index(halo_index);
          sod_halo_rdeltas(halo_index) =
              sorted_critical_bin_values(index).distance;
        });
      });

  Kokkos::Profiling::popRegion();

  return std::make_pair(sod_halo_rdeltas, sod_halo_rdeltas_index);
}

} // namespace Details

} // namespace ArborX

#endif
