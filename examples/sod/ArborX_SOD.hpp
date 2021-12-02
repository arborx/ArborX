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

namespace Details
{

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

    float dist = Details::distance(point, _fof_halo_centers(halo_index));
    if (dist > _r_max(halo_index)) // false positive
      return;

    auto bin_id = binID(_r_min, _r_max(halo_index), dist, _num_bins);
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
    Kokkos::atomic_fetch_add(&_sod_halo_bin_counts(halo_index, bin_id), 1);
    Kokkos::atomic_fetch_add(&_sod_halo_bin_masses(halo_index, bin_id),
                             _particle_masses(particle_index));
  }
};

template <typename MemorySpace>
struct SODParticlesCount
{
  Kokkos::View<int *, MemorySpace> _counts;

  template <typename Query>
  KOKKOS_FUNCTION void operator()(Query const &query, int halo_index) const
  {
    Kokkos::atomic_fetch_add(&_counts(halo_index), 1);
  }
};

struct SODTuple
{
  int particle_index;
  int halo_index;
  float distance;
  friend KOKKOS_FUNCTION bool operator<(SODTuple const &l, SODTuple const &r)
  {
    if (l.halo_index < r.halo_index)
      return true;
    if (l.halo_index > r.halo_index)
      return false;
    return l.distance <= r.distance;
  }
};

template <typename MemorySpace>
struct SODParticles
{
  Kokkos::View<int *, MemorySpace> _offsets;
  Kokkos::View<ArborX::Point *, MemorySpace> _fof_halo_centers;
  Kokkos::View<SODTuple *, MemorySpace> _values;

  template <typename Query>
  KOKKOS_FUNCTION void operator()(Query const &query, int halo_index) const
  {
    auto offset = Kokkos::atomic_fetch_add(&_offsets(halo_index), 1);

    int particle_index = getData(query);
    _values(offset).particle_index = particle_index;
    _values(offset).halo_index = halo_index;
    _values(offset).distance =
        distance(ArborX::getGeometry(query), _fof_halo_centers(halo_index));
  }
};

} // namespace Details

template <typename MemorySpace, typename Particles>
struct SODHandle
{
  Particles _particles;
  Kokkos::View<ArborX::Point *, MemorySpace> _fof_halo_centers;
  float _r_min;
  Kokkos::View<float *, MemorySpace> _r_max;
  BVH<MemorySpace> _bvh;

  template <typename ExecutionSpace>
  SODHandle(ExecutionSpace const &exec_space, Particles particles,
            Kokkos::View<ArborX::Point *, MemorySpace> fof_halo_centers,
            float r_min, Kokkos::View<float *, MemorySpace> r_max)
      : _particles(particles)
      , _fof_halo_centers(fof_halo_centers)
      , _r_min(r_min)
      , _r_max(r_max)
      , _bvh(exec_space, SOD::Spheres<MemorySpace>{fof_halo_centers, r_max})
  {
  }

  template <typename ExecutionSpace>
  void query(ExecutionSpace const &exec_space,
             Kokkos::View<int *, MemorySpace> &offsets,
             Kokkos::View<int *, MemorySpace> &indices) const
  {
    Kokkos::Profiling::pushRegion("ArborX::SODHandle::query");

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
        "ArborX::SODHandle::query::compute_particle_pairs");

    auto &counts = offsets; // alias
    Kokkos::deep_copy(counts, 0);
    _bvh.query(
        exec_space, SOD::ParticlesWrapper<Particles>{_particles},
        Details::SODParticlesCount<MemorySpace>{counts},
        Experimental::TraversalPolicy().setPredicateSorting(sort_predicates));

    exclusivePrefixSum(exec_space, offsets);

    auto const num_values = lastElement(offsets);
    printf("# values for all particles: %d\n", num_values);

    Kokkos::View<Details::SODTuple *, MemorySpace> values(
        "ArborX::SODHandle::query::values", num_values);
    auto offsets_clone = clone(offsets);
    _bvh.query(
        exec_space, SOD::ParticlesWrapper<Particles>{_particles},
        Details::SODParticles<MemorySpace>{offsets_clone, _fof_halo_centers,
                                           values},
        Experimental::TraversalPolicy().setPredicateSorting(sort_predicates));

    Kokkos::Profiling::popRegion();
    Kokkos::Profiling::pushRegion("ArborX::SODHandle::query::sort_pairs");

    // Sort
    // NOTE: current implementation also computes a permutation, which is
    // unnecessary
    sortObjects(exec_space, values);
    Kokkos::Profiling::popRegion();

    Kokkos::resize(indices, num_values);
    Kokkos::parallel_for(
        "ArborX::SODHandle::query::extract_indices",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_values),
        KOKKOS_LAMBDA(int const i) { indices(i) = values(i).particle_index; });

    Kokkos::Profiling::popRegion();
  }

  template <typename ExecutionSpace>
  void
  computeRdelta(ExecutionSpace const &exec_space,
                Kokkos::View<float *, MemorySpace> const &particle_masses,
                SOD::Parameters const &params,
                Kokkos::View<float *, MemorySpace> &sod_halo_rdeltas,
                Kokkos::View<int *, MemorySpace> &sod_halo_rdeltas_index) const
  {
    Kokkos::Profiling::pushRegion("ArborX::SODHandle::compute_R_delta");

    // TODO: for now, this is fixed to the usual number used for profiles.
    // But it does not have to. Need to play around with it to see what's the
    // fastest.
    constexpr int num_sod_bins = 20 + 1;

    auto const num_halos = _fof_halo_centers.extent(0);

    // Do not sort for now, so as to not allocate additional memory, which would
    // take 8*n bytes (4 for Morton index, 4 for permutation index). This will
    // results in running out of memory for the hardest HACC problem on V100.
    bool const sort_predicates = false;

    // Compute bin outer radii
    auto sod_halo_bin_outer_radii =
        Details::computeSODBinRadii(exec_space, _r_min, _r_max, num_sod_bins);

    // Step 2: compute some profiles (mass, count, avg radius);
    // NOTE: we will accumulate float quantities into double in order to
    // avoid loss of precision, which will occur once we start adding small
    // quantities to large
    Kokkos::View<double **, MemorySpace> sod_halo_bin_masses(
        "ArborX::SOD::sod_halo_bin_masses", num_halos, num_sod_bins);
    Kokkos::View<int **, MemorySpace> sod_halo_bin_counts(
        "ArborX::SOD::sod_halo_bin_masses", num_halos, num_sod_bins);
    computeBinProfiles(exec_space, num_sod_bins,
                       Details::MassAvgRadiiCountProfiles<MemorySpace>{
                           particle_masses, _fof_halo_centers,
                           sod_halo_bin_masses, sod_halo_bin_counts});

    // Figure out critical bins
    auto critical_bin_ids = Details::computeSODCriticalBins(
        exec_space, params, sod_halo_bin_masses, sod_halo_bin_counts,
        sod_halo_bin_outer_radii);

    Kokkos::Profiling::pushRegion(
        "ArborX::SOD::compute_critical_bin_particles");

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
    {
      auto offsets = clone(critical_bin_offsets);
      _bvh.query(
          exec_space, SOD::ParticlesWrapper<Particles>{_particles},
          Details::CriticalBinParticles<MemorySpace, Particles>{
              _particles, offsets, critical_bin_indices,
              critical_bin_distances_augmented, critical_bin_ids,
              _fof_halo_centers, sod_halo_bin_outer_radii, _r_min, _r_max},
          Experimental::TraversalPolicy().setPredicateSorting(sort_predicates));
    }

    Kokkos::Profiling::popRegion();

    {
      // Permute found particles within a critical bin of each halo based on
      // their distance to the center
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

    Kokkos::Profiling::pushRegion(
        "ArborX::SODHandle::compute_R_delta::compute_r_delta");
    // Compute R_delta
    Kokkos::resize(sod_halo_rdeltas_index, 0);
    std::tie(sod_halo_rdeltas, sod_halo_rdeltas_index) =
        Details::computeSODRdeltas(
            exec_space, params, particle_masses, sod_halo_bin_masses,
            sod_halo_bin_outer_radii, critical_bin_ids, critical_bin_offsets,
            critical_bin_indices, critical_bin_distances_augmented);
    Kokkos::Profiling::popRegion();

    Kokkos::parallel_for(
        "ArborX::SODHandle::compute_R_delta::update_rdeltas_index",
        Kokkos::RangePolicy<ExecutionSpace>(0, num_halos),
        KOKKOS_LAMBDA(int const halo_index) {
          for (int bin_id = 0; bin_id < critical_bin_ids(halo_index); ++bin_id)
            sod_halo_rdeltas_index(halo_index) +=
                sod_halo_bin_counts(halo_index, bin_id);
        });

    exec_space.fence();

    Kokkos::Profiling::popRegion();
  }

  template <typename ExecutionSpace>
  void computeRdeltaUsingScan(
      ExecutionSpace const &exec_space,
      Kokkos::View<float *, MemorySpace> const &particle_masses,
      SOD::Parameters const &params,
      Kokkos::View<float *, MemorySpace> &sod_halo_rdeltas,
      Kokkos::View<int *, MemorySpace> &sod_halo_rdeltas_index) const
  {
    Kokkos::Profiling::pushRegion(
        "ArborX::SODHandle::compute_R_delta_using_scan");

    auto const num_halos = _fof_halo_centers.extent(0);

    Kokkos::View<int *, MemorySpace> offsets;
    Kokkos::View<int *, MemorySpace> indices;
    query(exec_space, offsets, indices);

    using ParticlesAccess = ArborX::AccessTraits<Particles, PrimitivesTag>;

    auto particles = _particles;
    auto fof_halo_centers = _fof_halo_centers;

    using TeamPolicy =
        Kokkos::TeamPolicy<ExecutionSpace, Kokkos::Schedule<Kokkos::Dynamic>>;
    using team_member = typename TeamPolicy::member_type;

    ArborX::reallocWithoutInitializing(sod_halo_rdeltas, num_halos);
    ArborX::reallocWithoutInitializing(sod_halo_rdeltas_index, num_halos);
    Kokkos::deep_copy(sod_halo_rdeltas_index, INT_MAX);

    // Avoid capturing *this
    Kokkos::parallel_for(
        "ArborX::SODHandle::computeRdelta::compute_rdelta_index",
        TeamPolicy(num_halos, Kokkos::AUTO),
        KOKKOS_LAMBDA(const team_member &team) {
          auto halo_index = team.league_rank();

          auto halo_start = offsets(halo_index);
          auto num_points_in_halo = offsets(halo_index + 1) - halo_start;

          Kokkos::parallel_scan(
              Kokkos::TeamThreadRange(team, num_points_in_halo),
              [&](int i, double &accumulated_mass, bool const final_pass) {
                auto particle_index = indices(halo_start + i);

                accumulated_mass += particle_masses(particle_index);
                if (final_pass)
                {
                  auto r = Details::distance(
                      fof_halo_centers(halo_index),
                      ParticlesAccess::get(particles, particle_index));
                  float volume = 4.f / 3 * M_PI * pow(r, 3);
                  float ratio = (accumulated_mass / params._rho) / volume;

                  if (ratio <= params._rho_ratio)
                    Kokkos::atomic_min_fetch(
                        &sod_halo_rdeltas_index(halo_index), i);
                }
              });
        });

    Kokkos::parallel_for(
        "ArborX::SODHandle::computeRdelta::compute_rdelta",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_halos),
        KOKKOS_LAMBDA(int halo_index) {
          auto& rdelta_index = sod_halo_rdeltas_index(halo_index);
          if (rdelta_index < INT_MAX) {
              auto particle_index =
                  indices(offsets(halo_index) + rdelta_index);
              sod_halo_rdeltas(halo_index) = Details::distance(
                  fof_halo_centers(halo_index),
                  ParticlesAccess::get(particles, particle_index));
          }
        });

    exec_space.fence();

    Kokkos::Profiling::popRegion();
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
        exec_space, SOD::ParticlesWrapper<Particles>{_particles},
        Details::Profiles<MemorySpace, Callback>{_fof_halo_centers, _r_min,
                                                 _r_max, num_bins, callback},
        Experimental::TraversalPolicy().setPredicateSorting(sort_predicates));
  }
};

} // namespace ArborX

#endif
