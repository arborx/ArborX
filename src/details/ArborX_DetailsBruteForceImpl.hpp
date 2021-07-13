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

#ifndef ARBORX_DETAILS_BRUTE_FORCE_IMPL_HPP
#define ARBORX_DETAILS_BRUTE_FORCE_IMPL_HPP

#include <ArborX_AccessTraits.hpp>
#include <ArborX_DetailsAlgorithms.hpp> // expand
#include <ArborX_DetailsTreeConstruction.hpp> // Kokkos::reduction_identity<ArborX::Box>

#include <Kokkos_Core.hpp>

namespace ArborX
{
namespace Details
{
struct BruteForceImpl
{
  template <class ExecutionSpace, class Primitives, class BoundingVolumes,
            class Bounds>
  static void initializeBoundingVolumesAndReduceBoundsOfTheScene(
      ExecutionSpace const &space, Primitives const &primitives,
      BoundingVolumes const &bounding_volumes, Bounds &bounds)
  {
    using Access = AccessTraits<Primitives, PrimitivesTag>;

    int const n = Access::size(primitives);

    Kokkos::parallel_reduce("ArborX::BruteForce::BruteForce::"
                            "initialize_bounding_volumes_and_reduce_bounds",
                            Kokkos::RangePolicy<ExecutionSpace>(space, 0, n),
                            KOKKOS_LAMBDA(int i, Bounds &update) {
                              using Details::expand;
                              Bounds bounding_volume{};
                              expand(bounding_volume,
                                     Access::get(primitives, i));
                              bounding_volumes(i) = bounding_volume;
                              update += bounding_volume;
                            },
                            bounds);
  }

  template <class ExecutionSpace, class Primitives, class Predicates,
            class Callback>
  static void query(ExecutionSpace const &space, Primitives const &primitives,
                    Predicates const &predicates, Callback const &callback)
  {
    using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
    using AccessPrimitives = AccessTraits<Primitives, PrimitivesTag>;
    using AccessPredicates = AccessTraits<Predicates, PredicatesTag>;
    using PredicateType = typename AccessTraitsHelper<AccessPredicates>::type;
    using PrimitiveType = typename AccessTraitsHelper<AccessPrimitives>::type;

    int const n_primitives = AccessPrimitives::size(primitives);
    int const n_predicates = AccessPredicates::size(predicates);
    int max_scratch_size = TeamPolicy::scratch_size_max(0) / 2;
    int const predicates_per_team = max_scratch_size / sizeof(PredicateType);
    int const primitives_per_team = max_scratch_size / sizeof(PrimitiveType);

    int const n_primitive_tiles =
        ceil((float)n_primitives / primitives_per_team);
    int const n_predicate_tiles =
        ceil((float)n_predicates / predicates_per_team);
    int const n_teams = n_primitive_tiles * n_predicate_tiles;

    using ScratchPredicateType =
        Kokkos::View<PredicateType *,
                     typename ExecutionSpace::scratch_memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using ScratchPrimitiveType =
        Kokkos::View<PrimitiveType *,
                     typename ExecutionSpace::scratch_memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    int scratch_size = ScratchPredicateType::shmem_size(predicates_per_team) +
                       ScratchPrimitiveType::shmem_size(primitives_per_team);

    Kokkos::parallel_for(
        "ArborX::BruteForce::query::spatial::"
        "check_all_predicates_against_all_primitives",
        TeamPolicy((long)n_teams, Kokkos::AUTO)
            .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(const typename TeamPolicy::member_type &teamMember) {
          // select the tiles of predicates/primitives checked by each team
          int predicate_start = predicates_per_team *
                                (teamMember.league_rank() / n_primitive_tiles);
          int primitive_start = primitives_per_team *
                                (teamMember.league_rank() % n_primitive_tiles);

          int predicates_in_this_team = KokkosExt::min(
              predicates_per_team, n_predicates - predicate_start);
          int primitives_in_this_team = KokkosExt::min(
              primitives_per_team, n_primitives - primitive_start);

          ScratchPredicateType scratch_predicates(teamMember.team_scratch(0),
                                                  predicates_per_team);
          ScratchPrimitiveType scratch_primitives(teamMember.team_scratch(0),
                                                  primitives_per_team);
          // rank 0 in each team fills the scratch space with the
          // predicates / primitives in the tile
          if (teamMember.team_rank() == 0)
          {
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(teamMember,
                                          (long)predicates_in_this_team),
                [&](const int q) {
                  scratch_predicates(q) =
                      AccessPredicates::get(predicates, predicate_start + q);
                });
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(teamMember,
                                          (long)primitives_in_this_team),
                [&](const int j) {
                  scratch_primitives(j) =
                      AccessPrimitives::get(primitives, primitive_start + j);
                });
          }
          teamMember.team_barrier();

          // start threads for every predicate / primitive combination
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(teamMember,
                                      (long)primitives_in_this_team),
              [&](int j) {
                Kokkos::parallel_for(
                    Kokkos::ThreadVectorRange(teamMember,
                                              (long)predicates_in_this_team),
                    [&](const int q) {
                      auto const &predicate = scratch_predicates(q);
                      auto const &primitive = scratch_primitives(j);
                      if (predicate(primitive))
                      {
                        callback(predicate, j + primitive_start);
                      }
                    });
              });
        });
  }
};
} // namespace Details
} // namespace ArborX

#endif
