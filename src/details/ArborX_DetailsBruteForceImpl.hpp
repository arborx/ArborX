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

#ifndef ARBORX_DETAILS_BRUTE_FORCE_IMPL_HPP
#define ARBORX_DETAILS_BRUTE_FORCE_IMPL_HPP

#include <ArborX_DetailsAlgorithms.hpp> // expand
#include <ArborX_DetailsKokkosExtMinMaxOperations.hpp>
#include <ArborX_DetailsKokkosExtStdAlgorithms.hpp>
#include <ArborX_DetailsKokkosExtViewHelpers.hpp>
#include <ArborX_DetailsPriorityQueue.hpp>
#include <ArborX_Exception.hpp>

#include <Kokkos_Core.hpp>
#include <Kokkos_Profiling_ScopedRegion.hpp>

namespace ArborX::Details
{
struct BruteForceImpl
{
  template <class ExecutionSpace, class Values, class IndexableGetter,
            class Nodes, class Bounds>
  static void initializeBoundingVolumesAndReduceBoundsOfTheScene(
      ExecutionSpace const &space, Values const &values,
      IndexableGetter const &indexable_getter, Nodes const &nodes,
      Bounds &bounds)
  {
    Kokkos::parallel_reduce(
        "ArborX::BruteForce::BruteForce::"
        "initialize_values_and_reduce_bounds",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, values.size()),
        KOKKOS_LAMBDA(int i, Bounds &update) {
          nodes(i) = values(i);

          using Details::expand;
          Bounds bounding_volume{};
          expand(bounding_volume, indexable_getter(nodes(i)));
          update += bounding_volume;
        },
        Kokkos::Sum<Bounds>{bounds});
  }

  template <class ExecutionSpace, class Predicates, class Values,
            class Indexables, class Callback>
  static void query(SpatialPredicateTag, ExecutionSpace const &space,
                    Predicates const &predicates, Values const &values,
                    Indexables const &indexables, Callback const &callback)
  {
    Kokkos::Profiling::ScopedRegion guard("ArborX::BruteForce::query::spatial");

    using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
    using PredicateType = typename Predicates::value_type;
    using IndexableType = std::decay_t<decltype(indexables(0))>;

    int const n_indexables = values.size();
    int const n_predicates = predicates.size();
    int max_scratch_size = TeamPolicy::scratch_size_max(0);
    // half of the scratch memory used by predicates and half for indexables
    int const predicates_per_team =
        max_scratch_size / 2 / sizeof(PredicateType);
    int const indexables_per_team =
        max_scratch_size / 2 / sizeof(IndexableType);
    ARBORX_ASSERT(predicates_per_team > 0);
    ARBORX_ASSERT(indexables_per_team > 0);

    int const n_indexable_tiles =
        std::ceil((float)n_indexables / indexables_per_team);
    int const n_predicate_tiles =
        std::ceil((float)n_predicates / predicates_per_team);
    int const n_teams = n_indexable_tiles * n_predicate_tiles;

    using ScratchPredicateType =
        Kokkos::View<PredicateType *,
                     typename ExecutionSpace::scratch_memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using ScratchIndexableType =
        Kokkos::View<IndexableType *,
                     typename ExecutionSpace::scratch_memory_space,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    int scratch_size = ScratchPredicateType::shmem_size(predicates_per_team) +
                       ScratchIndexableType::shmem_size(indexables_per_team);

    Kokkos::parallel_for(
        "ArborX::BruteForce::query::spatial::"
        "check_all_predicates_against_all_indexables",
        TeamPolicy(space, n_teams, Kokkos::AUTO, 1)
            .set_scratch_size(0, Kokkos::PerTeam(scratch_size)),
        KOKKOS_LAMBDA(typename TeamPolicy::member_type const &teamMember) {
          // select the tiles of predicates/indexables checked by each team
          int predicate_start = predicates_per_team *
                                (teamMember.league_rank() / n_indexable_tiles);
          int indexable_start = indexables_per_team *
                                (teamMember.league_rank() % n_indexable_tiles);

          int predicates_in_this_team = KokkosExt::min(
              predicates_per_team, n_predicates - predicate_start);
          int indexables_in_this_team = KokkosExt::min(
              indexables_per_team, n_indexables - indexable_start);

          ScratchPredicateType scratch_predicates(teamMember.team_scratch(0),
                                                  predicates_per_team);
          ScratchIndexableType scratch_indexables(teamMember.team_scratch(0),
                                                  indexables_per_team);
          // fill the scratch space with the predicates / indexables in the
          // tile
          Kokkos::parallel_for(
              Kokkos::TeamVectorRange(teamMember, predicates_in_this_team),
              [&](const int q) {
                scratch_predicates(q) = predicates(predicate_start + q);
              });
          Kokkos::parallel_for(
              Kokkos::TeamVectorRange(teamMember, indexables_in_this_team),
              [&](const int j) {
                scratch_indexables(j) = indexables(indexable_start + j);
              });
          teamMember.team_barrier();

          // start threads for every predicate / indexable combination
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(teamMember, indexables_in_this_team),
              [&](int j) {
                Kokkos::parallel_for(
                    Kokkos::ThreadVectorRange(teamMember,
                                              predicates_in_this_team),
                    [&](const int q) {
                      auto const &predicate = scratch_predicates(q);
                      auto const &indexable = scratch_indexables(j);
                      if (predicate(indexable))
                      {
                        callback(predicate, values(indexable_start + j));
                      }
                    });
              });
        });
  }

  template <class ExecutionSpace, class Predicates, class Values,
            class Indexables, class Callback>
  static void query(NearestPredicateTag, ExecutionSpace const &space,
                    Predicates const &predicates, Values const &values,
                    Indexables const &indexables, Callback const &callback)
  {
    Kokkos::Profiling::ScopedRegion guard("ArborX::BruteForce::query::nearest");

    using MemorySpace = typename Values::memory_space;

    int const n_indexables = values.size();
    int const n_predicates = predicates.size();

    using Buffer = Kokkos::View<Kokkos::pair<int, float> *, MemorySpace>;
    using Offset = Kokkos::View<int *, MemorySpace>;
    struct BufferProvider
    {
      Buffer _buffer;
      Offset _offset;

      KOKKOS_FUNCTION auto operator()(int i) const
      {
        auto const *offset_ptr = &_offset(i);
        return Kokkos::subview(
            _buffer, Kokkos::make_pair(*offset_ptr, *(offset_ptr + 1)));
      }
    };

    Offset offset(
        Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                           "ArborX::BruteForce::query::nearest::offset"),
        n_predicates + 1);
    Kokkos::parallel_for(
        "ArborX::BruteForce::query::nearest::"
        "scan_queries_for_numbers_of_neighbors",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_predicates),
        KOKKOS_LAMBDA(int i) { offset(i) = getK(predicates(i)); });
    KokkosExt::exclusive_scan(space, offset, offset, 0);
    int const buffer_size = KokkosExt::lastElement(space, offset);

    Buffer buffer(Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                                     "ArborX::TreeTraversal::nearest::buffer"),
                  buffer_size);
    BufferProvider buffer_provider{buffer, offset};

    Kokkos::parallel_for(
        "ArborX::BruteForce::query::nearest::"
        "check_all_predicates_against_all_indexables",
        Kokkos::RangePolicy<ExecutionSpace>(space, 0, n_predicates),
        KOKKOS_LAMBDA(int i) {
          auto const &predicate = predicates(i);
          auto const k = getK(predicate);
          auto const buffer = buffer_provider(i);

          if (k < 1)
            return;

          auto radius = KokkosExt::ArithmeticTraits::infinity<float>::value;

          using PairIndexDistance = Kokkos::pair<int, float>;
          static_assert(std::is_same<typename decltype(buffer)::value_type,
                                     PairIndexDistance>::value,
                        "Type of the elements stored in the buffer passed as "
                        "argument to "
                        "TreeTraversal::nearestQuery is not right");
          struct CompareDistance
          {
            KOKKOS_INLINE_FUNCTION bool
            operator()(PairIndexDistance const &lhs,
                       PairIndexDistance const &rhs) const
            {
              return lhs.second < rhs.second;
            }
          };

          PriorityQueue<PairIndexDistance, CompareDistance,
                        UnmanagedStaticVector<PairIndexDistance>>
              heap(UnmanagedStaticVector<PairIndexDistance>(buffer.data(),
                                                            buffer.size()));

          for (int j = 0; j < n_indexables; ++j)
          {
            auto const distance = predicate.distance(indexables(j));
            if (distance < radius)
            {
              auto pair = Kokkos::make_pair(j, distance);
              if ((int)heap.size() < k)
                heap.push(pair);
              else
                heap.popPush(pair);
              if ((int)heap.size() == k)
                radius = heap.top().second;
            }
          }
          while (!heap.empty())
          {
            callback(predicate, values(heap.top().first));
            heap.pop();
          }
        });
  }
};

} // namespace ArborX::Details

#endif
