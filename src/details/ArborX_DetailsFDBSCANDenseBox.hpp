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

#ifndef ARBORX_DETAILSFDBSCANDENSEBOX_HPP
#define ARBORX_DETAILSFDBSCANDENSEBOX_HPP

#include <ArborX_Callbacks.hpp>
#include <ArborX_DetailsKokkosExtAccessibilityTraits.hpp>
#include <ArborX_DetailsKokkosExtViewHelpers.hpp>
#include <ArborX_DetailsUnionFind.hpp>
#include <ArborX_DetailsUtils.hpp>
#include <ArborX_Predicates.hpp>

#include <Kokkos_Core.hpp>

#include <cassert>

namespace ArborX
{
namespace Details
{

struct CartesianGrid
{
  Box _bounds;
  float _h;
  size_t _nx;
  size_t _ny;
  size_t _nz;

  CartesianGrid(Box const &bounds, float h)
      : _bounds(bounds)
      , _h(h)
  {
    auto const &min_corner = bounds.minCorner();
    auto const &max_corner = bounds.maxCorner();
    _nx = std::ceil((max_corner[0] - min_corner[0]) / h);
    _ny = std::ceil((max_corner[1] - min_corner[1]) / h);
    _nz = std::ceil((max_corner[2] - min_corner[2]) / h);

    // Catch potential overflow in grid cell indices early. This is a
    // conservative check as an actual overflow may not occur, depending on
    // which cells are filled.
    constexpr auto max_size_t = std::numeric_limits<size_t>::max();
    ARBORX_ASSERT(_nx == 0 || _ny == 0 || _nz == 0 ||
                  (_ny < max_size_t / _nx && _nz < max_size_t / (_nx * _ny)));
  }

  KOKKOS_FUNCTION
  size_t cellIndex(Point const &point) const
  {
    auto const &min_corner = _bounds.minCorner();
    size_t i = std::floor((point[0] - min_corner[0]) / _h);
    size_t j = std::floor((point[1] - min_corner[1]) / _h);
    size_t k = std::floor((point[2] - min_corner[2]) / _h);
    return k * _nx * _ny + j * _nx + i;
  }

  KOKKOS_FUNCTION
  Box cellBox(size_t cell_index) const
  {
    auto const &min_corner = _bounds.minCorner();

    auto i = cell_index % _nx;
    auto j = (cell_index / _nx) % _ny;
    auto k = cell_index / (_nx * _ny);
    return {{min_corner[0] + i * _h, min_corner[1] + j * _h,
             min_corner[2] + k * _h},
            {min_corner[0] + (i + 1) * _h, min_corner[1] + (j + 1) * _h,
             min_corner[2] + (k + 1) * _h}};
  }
};

template <typename MemorySpace, typename Primitives, typename DenseCellOffsets,
          typename Permutation>
struct CountUpToN_DenseBox
{
  Kokkos::View<int *, MemorySpace> _counts;
  Primitives _primitives;
  DenseCellOffsets _dense_cell_offsets;
  int _num_dense_cells;
  Permutation _permute;
  int core_min_size;
  float eps;
  int _n;

  CountUpToN_DenseBox(Kokkos::View<int *, MemorySpace> const &counts,
                      Primitives const &primitives,
                      DenseCellOffsets const &dense_cell_offsets,
                      Permutation const &permute, int core_min_size_in,
                      float eps_in, int n)
      : _counts(counts)
      , _primitives(primitives)
      , _dense_cell_offsets(dense_cell_offsets)
      , _num_dense_cells(dense_cell_offsets.size() - 1)
      , _permute(permute)
      , core_min_size(core_min_size_in)
      , eps(eps_in)
      , _n(n)
  {}

  template <typename Query>
  KOKKOS_FUNCTION auto operator()(Query const &query, int k) const
  {
    using Access = AccessTraits<Primitives, PrimitivesTag>;

    auto const i = getData(query);

    bool const is_dense_cell = (k < _num_dense_cells);

    int &count = _counts(i);
    if (is_dense_cell)
    {
      Point const &query_point = Access::get(_primitives, i);

      int const cell_start = _dense_cell_offsets(k);
      int const cell_end = _dense_cell_offsets(k + 1);
      for (int jj = cell_start; jj < cell_end; ++jj)
      {
        int j = _permute(jj);
        if (distance(query_point, Access::get(_primitives, j)) <= eps)
        {
          Kokkos::atomic_increment(&count);
          if (count >= _n)
            return ArborX::CallbackTreeTraversalControl::early_exit;
        }
      }
    }
    else
    {
      Kokkos::atomic_increment(&count);
      if (count >= _n)
        return ArborX::CallbackTreeTraversalControl::early_exit;
    }

    return ArborX::CallbackTreeTraversalControl::normal_continuation;
  }
};

template <typename MemorySpace, typename CorePointsType, typename Primitives,
          typename DenseCellOffsets, typename Permutation>
struct FDBSCANDenseBoxCallback
{
  UnionFind<MemorySpace> _union_find;
  CorePointsType _is_core_point;
  Primitives _primitives;
  DenseCellOffsets _dense_cell_offsets;
  int _num_dense_cells;
  int _num_points_in_dense_cells;
  Permutation _permute;
  float eps;

  template <typename ExecutionSpace>
  FDBSCANDenseBoxCallback(Kokkos::View<int *, MemorySpace> const &labels,
                          CorePointsType const &is_core_point,
                          Primitives const &primitives,
                          DenseCellOffsets const &dense_cell_offsets,
                          ExecutionSpace const &exec_space,
                          Permutation const &permute, float eps_in)
      : _union_find(labels)
      , _is_core_point(is_core_point)
      , _primitives(primitives)
      , _dense_cell_offsets(dense_cell_offsets)
      , _num_dense_cells(dense_cell_offsets.size() - 1)
      , _num_points_in_dense_cells(
            KokkosExt::lastElement(exec_space, _dense_cell_offsets))
      , _permute(permute)
      , eps(eps_in)
  {}

  template <typename Query>
  KOKKOS_FUNCTION auto operator()(Query const &query, int k) const
  {
    using Access = AccessTraits<Primitives, PrimitivesTag>;

    auto const i = ArborX::getData(query);

    bool const is_border_point = !_is_core_point(i);
    if (is_border_point)
      return ArborX::CallbackTreeTraversalControl::early_exit;

    bool const is_dense_cell = (k < _num_dense_cells);

    if (is_dense_cell)
    {
      int const cell_start = _dense_cell_offsets(k);
      int const cell_end = _dense_cell_offsets(k + 1);

      // Skip the dense box if they were already merged together
      if (_union_find.representative(i) ==
          _union_find.representative(_permute(cell_start)))
        return ArborX::CallbackTreeTraversalControl::normal_continuation;

      Point const &query_point = Access::get(_primitives, i);

      for (int jj = cell_start; jj < cell_end; ++jj)
      {
        int j = _permute(jj);

        // As soon as a pair is found, stop the search. If it is a case of
        // merging two dense cells, this will stop all other threads from
        // processing the same merge.
        if (_union_find.representative(i) == _union_find.representative(j))
          break;

        if (distance(query_point, Access::get(_primitives, j)) <= eps)
        {
          // We connected to at least one point in the dense box, thus we
          // connected to all of them, so may terminate
          _union_find.merge(i, j);
          break;
        }
      }
    }
    else
    {
      int j = _permute(_num_points_in_dense_cells + (k - _num_dense_cells));

      // No need to check the distance here, as the fact that we are inside the
      // callback guarantees that it is <= eps

      bool const is_neighbor_core_point = _is_core_point(j);
      if (is_neighbor_core_point && i > j)
        _union_find.merge(i, j);
      else if (!is_neighbor_core_point)
        _union_find.merge_into(j, i);
    }

    return ArborX::CallbackTreeTraversalControl::normal_continuation;
  }
};

template <typename ExecutionSpace, typename Primitives>
Kokkos::View<size_t *,
             typename AccessTraits<Primitives, PrimitivesTag>::memory_space>
computeCellIndices(ExecutionSpace const &exec_space,
                   Primitives const &primitives, CartesianGrid const &grid)
{
  using Access = AccessTraits<Primitives, PrimitivesTag>;
  using MemorySpace = typename Access::memory_space;

  auto const n = Access::size(primitives);

  Kokkos::View<size_t *, MemorySpace> cell_indices(
      Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing,
                         "ArborX::DBSCAN::cell_indices"),
      n);
  Kokkos::parallel_for(
      "ArborX::DBSCAN::compute_cell_indices",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
      KOKKOS_LAMBDA(int i) {
        auto const &xyz = Access::get(primitives, i);
        cell_indices(i) = grid.cellIndex(xyz);
      });
  return cell_indices;
}

template <typename ExecutionSpace, typename CellIndices, typename CellOffsets,
          typename Permutation>
int reorderDenseAndSparseCells(ExecutionSpace const &exec_space,
                               CellOffsets cell_offsets, int core_min_size,
                               CellIndices &sorted_cell_indices,
                               Permutation &permute)
{
  using MemorySpace = typename CellIndices::memory_space;

  static_assert(
      KokkosExt::is_accessible_from<MemorySpace, ExecutionSpace>::value, "");

  auto const num_nonempty_cells = cell_offsets.size() - 1;

  // Count the number of points in the dense cells
  int num_points_in_dense_cells = 0;
  Kokkos::parallel_reduce(
      "ArborX::DBSCAN::count_points_in_dense_cells",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_nonempty_cells),
      KOKKOS_LAMBDA(int i, int &update) {
        int num_points_in_cell = cell_offsets(i + 1) - cell_offsets(i);
        if (num_points_in_cell >= core_min_size)
          update += num_points_in_cell;
      },
      num_points_in_dense_cells);

  // Create new arrays of cells indices and permute, so that the points in the
  // dense cells go first, and the points in non-dense (sparse) cells go after
  // them. The points in the same cell are still together.
  Kokkos::View<int, MemorySpace> dense_offset("ArborX::DBSCAN::dense_offset");
  Kokkos::View<int, MemorySpace> sparse_offset("ArborX::DBSCAN::sparse_offset");
  Kokkos::deep_copy(exec_space, dense_offset, 0);
  Kokkos::deep_copy(exec_space, sparse_offset, num_points_in_dense_cells);

  auto reordered_permute =
      KokkosExt::cloneWithoutInitializingNorCopying(exec_space, permute);
  auto reordered_cell_indices = KokkosExt::cloneWithoutInitializingNorCopying(
      exec_space, sorted_cell_indices);
  Kokkos::parallel_for(
      "ArborX::DBSCAN::reorder_cell_indices_and_permutation",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_nonempty_cells),
      KOKKOS_LAMBDA(int i) {
        auto const num_points_in_cell = cell_offsets(i + 1) - cell_offsets(i);
        bool const is_dense_cell = (num_points_in_cell >= core_min_size);
        int offset = Kokkos::atomic_fetch_add(
            (is_dense_cell ? &dense_offset() : &sparse_offset()),
            num_points_in_cell);
        for (int j = cell_offsets(i); j < cell_offsets(i + 1); ++j, ++offset)
        {
          reordered_cell_indices(offset) = sorted_cell_indices(j);
          reordered_permute(offset) = permute(j);
        }
      });

  sorted_cell_indices = reordered_cell_indices;
  permute = reordered_permute;

  return num_points_in_dense_cells;
}

template <typename ExecutionSpace, typename CellIndices, typename Permutation,
          typename Labels>
void unionFindWithinEachDenseCell(ExecutionSpace const &exec_space,
                                  CellIndices sorted_dense_cell_indices,
                                  Permutation permute, Labels labels)
{
  using MemorySpace = typename Permutation::memory_space;

  UnionFind<MemorySpace> union_find{labels};

  // The algorithm relies on the fact that the cell indices array only contains
  // dense cells. Thus, as long as two cell indices are the same, a) they
  // belong to the same cell, and b) that cell is dense, thus they should be in
  // the same cluster. If, on the other hand, the array also contained
  // non-dense cells, that would not have been possible, as an additional
  // computations would have to be done to figure out if the points belong to a
  // dense cell, which would have required a linear scan.
  auto const n = sorted_dense_cell_indices.size();
  Kokkos::parallel_for(
      "ArborX::DBSCAN::union_find_within_each_dense_box",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 1, n),
      KOKKOS_LAMBDA(int i) {
        if (sorted_dense_cell_indices(i) == sorted_dense_cell_indices(i - 1))
          union_find.merge(permute(i), permute(i - 1));
      });
}

} // namespace Details
} // namespace ArborX

#endif
