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

#ifndef ARBORX_SPHERICAL_OVERDENSITY_HPP
#define ARBORX_SPHERICAL_OVERDENSITY_HPP

#include <ArborX.hpp>
#include <ArborX_DetailsSphericalOverdensity.hpp>

namespace ArborX
{

namespace SphericalOverdensity
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
} // namespace SphericalOverdensity

template <typename MemorySpace>
struct AccessTraits<SphericalOverdensity::Spheres<MemorySpace>, PrimitivesTag>
{
  using memory_space = MemorySpace;

  using Primitives = SphericalOverdensity::Spheres<MemorySpace>;

  KOKKOS_FUNCTION static std::size_t size(Primitives const &spheres)
  {
    return spheres._centers.extent(0);
  }
  KOKKOS_FUNCTION static Box get(Primitives const &spheres, std::size_t const i)
  {
    auto const &c = spheres._centers(i);
    auto const r = spheres._radii(i);
    return {{c[0] - r, c[1] - r, c[2] - r}, {c[0] + r, c[1] + r, c[2] + r}};
  }
};

template <typename Particles>
struct AccessTraits<SphericalOverdensity::ParticlesWrapper<Particles>,
                    PredicatesTag>
{
  using ParticlesAccess = AccessTraits<Particles, PrimitivesTag>;

  using memory_space = typename ParticlesAccess::memory_space;
  using Predicates = SphericalOverdensity::ParticlesWrapper<Particles>;

  static KOKKOS_FUNCTION size_t size(Predicates const &w)
  {
    return ParticlesAccess::size(w._particles);
  }
  static KOKKOS_FUNCTION auto get(Predicates const &w, size_t i)
  {
    return attach(intersects(ParticlesAccess::get(w._particles, i)), (int)i);
  }
};

template <typename MemorySpace, typename Particles>
struct SphericalOverdensityHandle
{
  Particles _particles;
  Kokkos::View<ArborX::Point *, MemorySpace> _fof_halo_centers;
  float _r_min;
  Kokkos::View<float *, MemorySpace> _r_max;
  BVH<MemorySpace> _bvh;

  template <typename ExecutionSpace>
  SphericalOverdensityHandle(
      ExecutionSpace const &exec_space, Particles particles,
      Kokkos::View<ArborX::Point *, MemorySpace> fof_halo_centers, float r_min,
      Kokkos::View<float *, MemorySpace> r_max)
      : _particles(particles)
      , _fof_halo_centers(fof_halo_centers)
      , _r_min(r_min)
      , _r_max(r_max)
      , _bvh(exec_space, SphericalOverdensity::Spheres<MemorySpace>{
                             fof_halo_centers, r_max})
  {
  }

  template <typename ExecutionSpace>
  void query(ExecutionSpace const &exec_space,
             Kokkos::View<int *, MemorySpace> &offsets,
             Kokkos::View<int *, MemorySpace> &indices) const
  {
    Kokkos::Profiling::pushRegion("ArborX::SOHandle::query");

    auto const num_halos = _fof_halo_centers.extent(0);

    // Do not sort for now, so as to not allocate additional memory, which would
    // take 8*n bytes (4 for Morton index, 4 for permutation index). This will
    // results in running out of memory for the hardest HACC problem on V100.
    bool const sort_predicates = false;

    Kokkos::resize(offsets, num_halos + 1);

    // We do two passes (count and fill) here. However, we can't use the ArborX
    // core for that as what we fill is not associated with each query, but
    // rather with each halo.

    Kokkos::Profiling::pushRegion(
        "ArborX::SOHandle::query::compute_particle_pairs");

    auto &counts = offsets; // alias
    Kokkos::deep_copy(exec_space, counts, 0);
    _bvh.query(
        exec_space,
        SphericalOverdensity::ParticlesWrapper<Particles>{_particles},
        Details::SOParticlesCount<MemorySpace>{_fof_halo_centers, _r_max,
                                               counts},
        Experimental::TraversalPolicy().setPredicateSorting(sort_predicates));

    exclusivePrefixSum(exec_space, offsets);

    auto const num_values = lastElement(offsets);
    printf("# values for all particles: %d\n", num_values);

    Kokkos::View<Details::SOTuple *, MemorySpace> values(
        "ArborX::SOHandle::query::values", num_values);
    auto offsets_clone = clone(exec_space, offsets);
    _bvh.query(
        exec_space,
        SphericalOverdensity::ParticlesWrapper<Particles>{_particles},
        Details::SOParticles<MemorySpace>{_fof_halo_centers, _r_max,
                                          offsets_clone, values},
        Experimental::TraversalPolicy().setPredicateSorting(sort_predicates));

    Kokkos::Profiling::popRegion();
    Kokkos::Profiling::pushRegion("ArborX::SOHandle::query::sort_values");

    // Sort
    sortObjects(exec_space, values);
    Kokkos::Profiling::popRegion();

    Kokkos::resize(indices, num_values);
    Kokkos::parallel_for(
        "ArborX::SOHandle::query::extract_indices",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_values),
        KOKKOS_LAMBDA(int const i) { indices(i) = values(i).particle_index; });

    Kokkos::Profiling::popRegion();
  }

  template <typename ExecutionSpace>
  void computeRdelta(ExecutionSpace const &exec_space,
                     Kokkos::View<float *, MemorySpace> const &particle_masses,
                     SphericalOverdensity::Parameters const &params,
                     Kokkos::View<float *, MemorySpace> &sod_halo_rdeltas,
                     Kokkos::View<int *, MemorySpace> &sod_halo_rdeltas_index,
                     bool use_bin_approach = true) const
  {
    auto const num_halos = _fof_halo_centers.extent(0);

    auto rho = params._rho;
    auto rho_ratio = params._rho_ratio;

    if (use_bin_approach)
    {
      Kokkos::Profiling::pushRegion(
          "ArborX::SOHandle::compute_R_delta_with_bins");

      // TODO: for now, this is fixed to the usual number used for profiles.
      // But it does not have to. Need to play around with it to see what's the
      // fastest.
      constexpr int num_sod_bins = 20 + 1;

      // Do not sort for now, so as to not allocate additional memory, which
      // would take 8*n bytes (4 for Morton index, 4 for permutation index).
      // This will results in running out of memory for the hardest HACC problem
      // on V100.
      bool const sort_predicates = false;

      // Compute bin outer radii
      auto sod_halo_bin_outer_radii =
          Details::computeSOBinRadii(exec_space, _r_min, _r_max, num_sod_bins);

      // Step 2: compute some profiles (mass, count, avg radius);
      // NOTE: we will accumulate float quantities into double in order to
      // avoid loss of precision, which will occur once we start adding small
      // quantities to large
      Kokkos::View<double **, MemorySpace> sod_halo_bin_masses(
          "ArborX::SO::sod_halo_bin_masses", num_halos, num_sod_bins);
      Kokkos::View<int **, MemorySpace> sod_halo_bin_counts(
          "ArborX::SO::sod_halo_bin_masses", num_halos, num_sod_bins);
      computeBinProfiles(exec_space, num_sod_bins,
                         Details::MassAvgRadiiCountProfiles<MemorySpace>{
                             particle_masses, _fof_halo_centers,
                             sod_halo_bin_masses, sod_halo_bin_counts});

      // Figure out critical bins
      auto critical_bin_ids = Details::computeSOCriticalBins(
          exec_space, rho, rho_ratio, sod_halo_bin_masses, sod_halo_bin_counts,
          sod_halo_bin_outer_radii);

      Kokkos::Profiling::pushRegion(
          "ArborX::SO::compute_critical_bin_particles");

      // Compute offsets for storing particles in critical bins
      Kokkos::View<int *, MemorySpace> critical_bin_offsets(
          Kokkos::view_alloc(Kokkos::WithoutInitializing,
                             "ArborX::SO::critical_bin_offsets"),
          num_halos + 1);
      Kokkos::parallel_for(
          "ArborX::SO::compute_critical_bins",
          Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_halos),
          KOKKOS_LAMBDA(int halo_index) {
            critical_bin_offsets(halo_index) =
                sod_halo_bin_counts(halo_index, critical_bin_ids(halo_index));
          });
      exclusivePrefixSum(exec_space, critical_bin_offsets);

      auto num_critical_bin_particles = lastElement(critical_bin_offsets);
      printf("#particles in critical bins: %d\n", num_critical_bin_particles);

      // Find particles in critical bins for each halo
      Kokkos::View<Details::SOTuple *, MemorySpace> critical_bin_values(
          Kokkos::view_alloc(Kokkos::WithoutInitializing,
                             "ArborX::SO::critical_bin_values"),
          num_critical_bin_particles);
      {
        auto offsets = clone(exec_space, critical_bin_offsets);
        _bvh.query(
            exec_space,
            SphericalOverdensity::ParticlesWrapper<Particles>{_particles},
            Details::CriticalBinParticles<MemorySpace, Particles>{
                _particles, critical_bin_ids, _fof_halo_centers, num_sod_bins,
                _r_min, _r_max, offsets, critical_bin_values},
            Experimental::TraversalPolicy().setPredicateSorting(
                sort_predicates));
      }

      Kokkos::Profiling::popRegion();

      // Sort the particles based on their distance to the corresponding FOF
      // center
      Kokkos::Profiling::pushRegion(
          "ArborX::SOHandle::computeRdelta::sort_values");
      sortObjects(exec_space, critical_bin_values);
      Kokkos::Profiling::popRegion();

      // Compute R_delta
      Kokkos::Profiling::pushRegion(
          "ArborX::SOHandle::compute_R_delta::compute_r_delta");
      Kokkos::resize(sod_halo_rdeltas_index, 0);
      std::tie(sod_halo_rdeltas, sod_halo_rdeltas_index) =
          Details::computeSORdeltas(exec_space, rho, rho_ratio, particle_masses,
                                    sod_halo_bin_masses, critical_bin_ids,
                                    critical_bin_offsets, critical_bin_values);
      Kokkos::Profiling::popRegion();

      Kokkos::parallel_for(
          "ArborX::SOHandle::compute_R_delta::update_rdeltas_index",
          Kokkos::RangePolicy<ExecutionSpace>(0, num_halos),
          KOKKOS_LAMBDA(int const halo_index) {
            for (int bin_id = 0; bin_id < critical_bin_ids(halo_index);
                 ++bin_id)
              sod_halo_rdeltas_index(halo_index) +=
                  sod_halo_bin_counts(halo_index, bin_id);
          });

      Kokkos::Profiling::popRegion();
    }
    else
    {
      Kokkos::Profiling::pushRegion(
          "ArborX::SOHandle::compute_R_delta_no_bins");

      Kokkos::View<int *, MemorySpace> offsets;
      Kokkos::View<int *, MemorySpace> indices;
      query(exec_space, offsets, indices);

      using ParticlesAccess = ArborX::AccessTraits<Particles, PrimitivesTag>;

      using TeamPolicy =
          Kokkos::TeamPolicy<ExecutionSpace, Kokkos::Schedule<Kokkos::Dynamic>>;
      using team_member = typename TeamPolicy::member_type;

      constexpr int invalid = INT_MAX;
      ArborX::reallocWithoutInitializing(sod_halo_rdeltas, num_halos);
      ArborX::reallocWithoutInitializing(sod_halo_rdeltas_index, num_halos);
      Kokkos::deep_copy(sod_halo_rdeltas_index, invalid);

      // Avoid capturing *this
      auto const &particles = _particles;
      auto const &fof_halo_centers = _fof_halo_centers;

      Kokkos::parallel_for(
          "ArborX::SOHandle::computeRdelta::compute_rdelta_index",
          TeamPolicy(exec_space, num_halos, 512),
          KOKKOS_LAMBDA(const team_member &team) {
            auto halo_index = team.league_rank();

            auto halo_start = offsets(halo_index);
            auto num_points_in_halo = offsets(halo_index + 1) - halo_start;

            Kokkos::parallel_scan(
                Kokkos::TeamThreadRange(team, num_points_in_halo),
                [&](int i, double &accumulated_mass, bool const final_pass) {
                  auto particle_index = indices(halo_start + i);

                  // FIXME: Adding numbers of different orders of magnitude
                  accumulated_mass += particle_masses(particle_index);
                  if (final_pass)
                  {
                    auto r = Details::distance(
                        fof_halo_centers(halo_index),
                        ParticlesAccess::get(particles, particle_index));
                    float volume = 4.f / 3 * M_PI * pow(r, 3);
                    float ratio = (accumulated_mass / rho) / volume;

                    if (ratio <= rho_ratio)
                      Kokkos::atomic_min(&sod_halo_rdeltas_index(halo_index),
                                         i);
                  }
                });
          });

      Kokkos::parallel_for(
          "ArborX::SOHandle::computeRdelta::compute_rdelta",
          Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_halos),
          KOKKOS_LAMBDA(int halo_index) {
            auto &rdelta_index = sod_halo_rdeltas_index(halo_index);
            if (rdelta_index == invalid)
            {
              // If we never found a particle satisfying the threshold
              // criteria, assign the index to the last particle in the halo.
              auto const num_points_in_halo =
                  offsets(halo_index + 1) - offsets(halo_index);
              assert(num_points_in_halo > 0);
              rdelta_index = num_points_in_halo - 1;
            }

            auto particle_index = indices(offsets(halo_index) + rdelta_index);
            sod_halo_rdeltas(halo_index) = Details::distance(
                fof_halo_centers(halo_index),
                ParticlesAccess::get(particles, particle_index));
          });

      Kokkos::Profiling::popRegion();
    }
    exec_space.fence();
  }

  template <typename ExecutionSpace, typename Callback>
  void computeBinProfiles(ExecutionSpace const &exec_space, int num_bins,
                          Callback const &callback) const
  {
    // Do not sort for now, so as to not allocate additional memory, which
    // would take 8*n bytes (4 for Morton index, 4 for permutation index).
    // This will results in running out of memory for the hardest HACC problem
    // on V100.
    bool const sort_predicates = false;
    _bvh.query(
        exec_space,
        SphericalOverdensity::ParticlesWrapper<Particles>{_particles},
        Details::Profiles<MemorySpace, Callback>{_fof_halo_centers, _r_min,
                                                 _r_max, num_bins, callback},
        Experimental::TraversalPolicy().setPredicateSorting(sort_predicates));
  }
};

} // namespace ArborX

#endif
