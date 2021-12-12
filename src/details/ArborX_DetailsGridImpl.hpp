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
  struct Tuple3
  {
    size_t i;
    size_t j;
    size_t k;
  };

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

  template <class ExecutionSpace, typename BinOffsets1D, typename Indices>
  static Kokkos::View<int *, typename Indices::memory_space>
  computeBinIndices(ExecutionSpace const &exec_space,
                    BinOffsets1D const &bin_offsets_1d,
                    Indices const &sorted_indices)
  {
    auto const num_bins = bin_offsets_1d.extent(0) - 1;
    Kokkos::View<int *, typename Indices::memory_space> bin_indices_1d(
        Kokkos::view_alloc(Kokkos::WithoutInitializing,
                           "ArborX::Grid::Grid::bin_indices_1d"),
        num_bins);
    Kokkos::parallel_for(
        "ArborX::Grid::Grid::compute_bin_indices_1d",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_bins),
        KOKKOS_LAMBDA(int const i) {
          bin_indices_1d(i) = sorted_indices(bin_offsets_1d(i));
        });
    return bin_indices_1d;
  }

  template <class ExecutionSpace, typename BinOffsets1D, typename BinIndices1D,
            typename Bins3DHash>
  static void convertBinOffsetsTo3D(ExecutionSpace const &exec_space,
                                    Details::CartesianGrid const &grid,
                                    BinOffsets1D const &bin_offsets_1d,
                                    BinIndices1D const &bin_indices_1d,
                                    Bins3DHash &bins_3d_hash)
  {
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

          auto const bin_offset = bin_offsets_1d(index);
          auto const bin_count =
              bin_offsets_1d(index + 1) - bin_offsets_1d(index);
          bins_3d_hash.insert(Tuple3{i, j, k},
                              Kokkos::make_pair(bin_offset, bin_count));
        });
  }

  template <class ExecutionSpace, class Predicates, class Callback,
            typename Permute, class Points, typename Bins3DHash>
  static void
  query(ExecutionSpace const &exec_space, Predicates const &predicates,
        Callback const &callback, Permute const &permute, Points const &points,
        Details::CartesianGrid const &grid, Bins3DHash const &bins_3d_hash)
  {
    using Access = AccessTraits<Predicates, PredicatesTag>;

    auto num_predicates = Access::size(predicates);
    Kokkos::parallel_for(
        "ArborX::Grid::query::spatial",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_predicates),
        KOKKOS_LAMBDA(int const i) {
          auto const &predicate = Access::get(predicates, i);

          size_t bin_index = grid.cellIndex(getGeometry(predicate).centroid());

          size_t bin_i;
          size_t bin_j;
          size_t bin_k;
          grid.cellIndex2Triplet(bin_index, bin_i, bin_j, bin_k);

          using KokkosExt::min;
          for (size_t bk = min(bin_k - 1, (size_t)0);
               bk <= min(bin_k + 1, grid._nz - 1); ++bk)
            for (size_t bj = min(bin_j - 1, (size_t)0);
                 bj <= min(bin_j + 1, grid._ny - 1); ++bj)
              for (size_t bi = min(bin_i - 1, (size_t)0);
                   bi <= min(bin_i + 1, grid._nx - 1); ++bi)
              {
                auto map_index = bins_3d_hash.find(Tuple3{bi, bj, bk});
                if (!bins_3d_hash.valid_at(map_index))
                  continue;

                auto const &bin_data = bins_3d_hash.value_at(map_index);
                auto neigh_bin_offset = bin_data.first;
                auto num_neigh_bin_points = bin_data.second;

                for (int jj = 0; jj < num_neigh_bin_points; ++jj)
                {
                  auto const j = neigh_bin_offset + jj;
                  if (predicate(points(j)))
                    callback(predicate, permute(j));
                }
              }
        });
  }
};

} // namespace Details
} // namespace ArborX

#endif
