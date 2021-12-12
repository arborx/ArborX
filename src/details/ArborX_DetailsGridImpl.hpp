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

#include <ArborX_AccessTraits.hpp>
#include <ArborX_DetailsAlgorithms.hpp> // expand
#include <ArborX_DetailsTreeConstruction.hpp> // Kokkos::reduction_identity<ArborX::Point>

#include <Kokkos_Core.hpp>

namespace ArborX
{
namespace Details
{
struct GridImpl
{

  template <typename ExecutionSpace, typename Primitives>
  inline static void
  calculateBoundingBoxOfTheScene(ExecutionSpace const &space,
                                 Primitives const &primitives,
                                 Box &scene_bounding_box)
  {
    using Access = AccessTraits<Primitives, PrimitivesTag>;
    auto const n = Access::size(primitives);
    Kokkos::parallel_reduce("ArborX::Grid::calculate_bounding_box_of_the_scene",
                            Kokkos::RangePolicy<ExecutionSpace>(space, 0, n),
                            KOKKOS_LAMBDA(int i, Box &update) {
                              update += Access::get(primitives, i);
                            },
                            scene_bounding_box);
  }

  template <class ExecutionSpace, class Primitives, typename Permute,
            typename Points>
  static void initializePoints(ExecutionSpace const &exec_space,
                               Primitives const &primitives,
                               Permute const &permute, Points &points)
  {
    using Access = AccessTraits<Primitives, PrimitivesTag>;

    int const n = Access::size(primitives);
    Kokkos::parallel_for("ArborX::Grid::Grid::initialize_points",
                         Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
                         KOKKOS_LAMBDA(int i) {
                           points(i) = Access::get(primitives, permute(i));
                         });
  }

  template <class ExecutionSpace, typename BinOffsets1D, typename Indices,
            typename BinIndices1D>
  static void computeBinIndices(ExecutionSpace const &exec_space,
                                BinOffsets1D const &bin_offsets_1d,
                                Indices const &sorted_indices,
                                BinIndices1D &bin_indices_1d)
  {
    auto const num_bins = bin_offsets_1d.extent(0) - 1;
    Kokkos::resize(Kokkos::WithoutInitializing, bin_indices_1d, num_bins);
    Kokkos::parallel_for(
        "ArborX::Grid::Grid::compute_bin_indices_1d",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_bins),
        KOKKOS_LAMBDA(int const i) {
          bin_indices_1d(i) = sorted_indices(bin_offsets_1d(i));
        });
  }

  template <class ExecutionSpace, typename BinOffsets1D, typename BinIndices1D,
            typename BinOffsets3D, typename BinCounts3D>
  static void convertBinOffsetsTo3D(ExecutionSpace const &exec_space,
                                    Details::CartesianGrid const &grid,
                                    BinOffsets1D const &bin_offsets_1d,
                                    BinIndices1D const &bin_indices_1d,
                                    BinOffsets3D &bin_offsets_3d,
                                    BinCounts3D &bin_counts_3d)
  {
    Kokkos::resize(Kokkos::WithoutInitializing, bin_offsets_3d, grid._nx,
                   grid._ny, grid._nz);
    Kokkos::deep_copy(bin_offsets_3d, -1);
    Kokkos::resize(bin_counts_3d, grid._nx, grid._ny, grid._nz);

    auto const num_bins = bin_offsets_1d.extent_int(0) - 1;
    Kokkos::parallel_for(
        "ArborX::Grid::Grid::compute_bin_offset_3d_mapping",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_bins),
        KOKKOS_LAMBDA(int index) {
          size_t bin_index = bin_indices_1d(index);
          size_t i;
          size_t j;
          size_t k;
          grid.cellIndex2Triplet(bin_index, i, j, k);
          bin_offsets_3d(i, j, k) = bin_offsets_1d(index);
          bin_counts_3d(i, j, k) =
              bin_offsets_1d(index + 1) - bin_offsets_1d(index);
        });
  }

  template <class ExecutionSpace, class Predicates, class Callback,
            typename Permute, class Points, typename BinIndices1D,
            typename BinOffsets3D, typename BinCounts3D>
  static void
  query(ExecutionSpace const &exec_space, Predicates const &predicates,
        Callback const &callback, Permute const &permute, Points const &points,
        Details::CartesianGrid const &grid, BinIndices1D const &bin_indices_1d,
        BinOffsets3D const &bin_offsets_3d, BinCounts3D const &bin_counts_3d)
  {
    using Access = AccessTraits<Predicates, PredicatesTag>;

    using TeamPolicy =
        Kokkos::TeamPolicy<ExecutionSpace, Kokkos::Schedule<Kokkos::Dynamic>>;
    using team_member = typename TeamPolicy::member_type;

    auto const num_bins = bin_indices_1d.extent(0);
    Kokkos::parallel_for(
        "ArborX::Grid::query::spatial",
        TeamPolicy(exec_space, num_bins, Kokkos::AUTO, 8),
        KOKKOS_LAMBDA(team_member const &team) {
          auto index = team.league_rank();
          size_t bin_index = bin_indices_1d(index);

          size_t bin_i;
          size_t bin_j;
          size_t bin_k;
          grid.cellIndex2Triplet(bin_index, bin_i, bin_j, bin_k);

          auto const bin_offset = bin_offsets_3d(bin_i, bin_j, bin_k);
          auto const num_bin_points = bin_counts_3d(bin_i, bin_j, bin_k);
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, num_bin_points), [&](int const ii) {
                auto const i = permute(bin_offset + ii);

                auto const &predicate = Access::get(predicates, i);

                using KokkosExt::min;
                for (size_t bk = min(bin_k - 2, (size_t)0);
                     bk <= min(bin_k + 2, grid._nz - 1); ++bk)
                  for (size_t bj = min(bin_j - 2, (size_t)0);
                       bj <= min(bin_j + 2, grid._ny - 1); ++bj)
                    for (size_t bi = min(bin_i - 2, (size_t)0);
                         bi <= min(bin_i + 2, grid._nx - 1); ++bi)
                    {
                      auto neigh_bin_offset = bin_offsets_3d(bi, bj, bk);
                      auto num_neigh_bin_points = bin_counts_3d(bi, bj, bk);

                      Kokkos::parallel_for(
                          Kokkos::ThreadVectorRange(team, num_neigh_bin_points),
                          [&](int const jj) {
                            auto const j = neigh_bin_offset + jj;
                            if (predicate(points(j)))
                              callback(predicate, permute(j));
                          });
                    }
              });
        });
  }
};

} // namespace Details
} // namespace ArborX

#endif
