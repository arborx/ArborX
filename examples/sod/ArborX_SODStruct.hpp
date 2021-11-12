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

#ifndef ARBORX_DETAILSSODSTRUCT_HPP
#define ARBORX_DETAILSSODSTRUCT_HPP

#include <ArborX.hpp>

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

namespace SOD
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
} // namespace SOD

template <typename MemorySpace>
struct SODStruct
{
  Kokkos::View<ArborX::Point *, MemorySpace> _fof_halo_centers;
  float _r_min;
  Kokkos::View<float *, MemorySpace> _r_max;
  BVH<MemorySpace> _bvh;

  template <typename ExecutionSpace>
  SODStruct(ExecutionSpace const &exec_space,
            Kokkos::View<ArborX::Point *, MemorySpace> fof_halo_centers,
            float r_min, Kokkos::View<float *, MemorySpace> r_max)
      : _fof_halo_centers(fof_halo_centers)
      , _r_min(r_min)
      , _r_max(r_max)
      , _bvh(exec_space, SOD::Spheres<MemorySpace>{fof_halo_centers, r_max})
  {
  }

#if 0
  template <typename ExecutionSpace>
  void query(ExecutionSpace const &exec_space,
             Kokkos::View<ArborX::Point *, MemorySpace> const &particles,
             Kokkos::View<int *, MemorySpace> &offsets,
             Kokkos::View<int *, MemorySpace> &indices) const
  {
    bvh.query(exec_space, ParticlesWrapper<Particles>{particles},
              SODParticles<MemorySpace, Particles>{offsets, indices});
    // TODO: sort results based on distance
  }
#endif

#if 0
  template <typename ExecutionSpace>
  void
  computeRdelta(ExectuionSpace const &exec_space,
                Kokkos::View<ArborX::Point *, MemorySpace> const &particles,
                Kokkos::View<float *, MemorySpace> const &particle_masses,
                Kokkos::View<float *, MemorySpace> &r_delta) const
  {
    constexpr num_sod_bins = 20 + 1;
    // Do not sort for now, so as to not allocate additional memory, which would
    // take 8*n bytes (4 for Morton index, 4 for permutation index). This will
    // results in running out of memory for the hardest HACC problem on V100.
    bool const sort_predicates = false;

    // Compute bin outer radii
    auto sod_halo_bin_outer_radii =
        computeSODBinRadii(exec_space, _r_min, _r_max, num_sod_bins);

    // Step 2: compute some profiles (mass, count, avg radius);
    // NOTE: we will accumulate float quantities into double in order to
    // avoid loss of precision, which will occur once we start adding small
    // quantities to large
    Kokkos::View<double **, MemorySpace> sod_halo_bin_masses(
        "ArborX::SOD::sod_halo_bin_masses", num_halos, num_sod_bins);
    Kokkos::View<int **, MemorySpace> sod_halo_bin_counts(
        "ArborX::SOD::sod_halo_bin_masses", num_halos, num_sod_bins);
    Kokkos::resize(out.sod_halo_bin_counts, num_halos, num_sod_bins);
    bvh.query(
        exec_space, ParticlesWrapper<Particles>{particles},
        BinAccumulator<MemorySpace, Particles>{
            particles, particle_masses, sod_halo_bin_counts,
            sod_halo_bin_masses, fof_halo_centers, r_min, r_max},
        Experimental::TraversalPolicy().setPredicateSorting(sort_predicates));

    // Figure out critical bins
    auto critical_bin_ids =
        computeSODCriticalBins(exec_space, params, sod_halo_bin_masses,
                               sod_halo_bin_counts, sod_halo_bin_outer_radii);

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

    // Compute R_delta
    Kokkos::View<int *, MemorySpace> sod_halo_rdeltas_index(
        "ArborX::SOD::r_delta_index", 0);
    std::tie(out.sod_halo_rdeltas, sod_halo_rdeltas_index) = computeSODRdeltas(
        exec_space, params, particle_masses, sod_halo_bin_masses,
        sod_halo_bin_outer_radii, critical_bin_ids, critical_bin_offsets,
        critical_bin_indices, critical_bin_distances_augmented);

    Kokkos::Profiling::pushRegion("ArborX::SOD::find_sod_particles");
  }
#endif

  template <typename ExecutionSpace, typename Particles, typename Callback>
  void computeBinProfiles(ExecutionSpace const &exec_space,
                          Particles const &particles, int num_bins,
                          Callback const &callback) const
  {
    // Do not sort for now, so as to not allocate additional memory, which would
    // take 8*n bytes (4 for Morton index, 4 for permutation index). This will
    // results in running out of memory for the hardest HACC problem on V100.
    bool const sort_predicates = false;
    _bvh.query(
        exec_space, SOD::ParticlesWrapper<Particles>{particles},
        SOD::Profiles<MemorySpace, Callback>{_fof_halo_centers, _r_min, _r_max,
                                             num_bins, callback},
        Experimental::TraversalPolicy().setPredicateSorting(sort_predicates));
  }
};

} // namespace ArborX

#endif
