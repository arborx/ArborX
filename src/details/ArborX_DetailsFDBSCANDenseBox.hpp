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

#ifndef ARBORX_DETAILSFDBSCANDENSEBOX_HPP
#define ARBORX_DETAILSFDBSCANDENSEBOX_HPP

#include <ArborX_Callbacks.hpp>
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

template <typename MemorySpace, typename Primitives, typename MixedOffsets,
          typename Permutation>
struct FDBSCANDenseBoxCorePointsCallback
{
  Kokkos::View<int *, MemorySpace> _num_neigh;
  Primitives _primitives;
  MixedOffsets _mixed_offsets;
  Permutation _permute;
  int core_min_size;
  float eps;

  template <typename Query>
  KOKKOS_FUNCTION auto operator()(Query const &query, int k) const
  {
    using Access = AccessTraits<Primitives, PrimitivesTag>;

    auto i = getData(query);

    bool is_dense_cell =
        (_mixed_offsets(k + 1) - _mixed_offsets(k) >= core_min_size);

    if (is_dense_cell)
    {
      for (int jj = _mixed_offsets(k); jj < _mixed_offsets(k + 1); ++jj)
      {
        int j = _permute(jj);
        if (distance(Access::get(_primitives, i),
                     Access::get(_primitives, j)) <= eps)
        {
          Kokkos::atomic_fetch_add(&_num_neigh(i), 1);

          if (_num_neigh(i) >= core_min_size)
            return ArborX::CallbackTreeTraversalControl::early_exit;
        }
      }
    }
    else
    {
      assert(_mixed_offsets(k + 1) - _mixed_offsets(k) == 1);

      Kokkos::atomic_fetch_add(&_num_neigh(i), 1);

      if (_num_neigh(i) >= core_min_size)
        return ArborX::CallbackTreeTraversalControl::early_exit;
    }

    return ArborX::CallbackTreeTraversalControl::normal_continuation;
  }
};

template <typename MemorySpace, typename CorePointsType, typename Primitives,
          typename MixedOffsets, typename Point2Offset, typename Permutation>
struct FDBSCANDenseBoxCallback
{
  UnionFind<MemorySpace> _union_find;
  CorePointsType _is_core_point;
  Primitives _primitives;
  MixedOffsets _mixed_offsets;
  Point2Offset _point2offset;
  Permutation _permute;
  int core_min_size;
  float eps;

  FDBSCANDenseBoxCallback(Kokkos::View<int *, MemorySpace> const &labels,
                          CorePointsType const &is_core_point,
                          Primitives const &primitives,
                          MixedOffsets const &mixed_offsets,
                          Point2Offset const &point2offset,
                          Permutation const &permute, int core_min_size_in,
                          float eps_in)
      : _union_find(labels)
      , _is_core_point(is_core_point)
      , _primitives(primitives)
      , _mixed_offsets(mixed_offsets)
      , _point2offset(point2offset)
      , _permute(permute)
      , core_min_size(core_min_size_in)
      , eps(eps_in)
  {
  }

  template <typename Query>
  KOKKOS_FUNCTION auto operator()(Query const &query, int k) const
  {
    using Access = AccessTraits<Primitives, PrimitivesTag>;

    int const i = ArborX::getData(query);

    bool is_boundary_point = !_is_core_point(i);
    bool is_dense_cell =
        (_mixed_offsets(k + 1) - _mixed_offsets(k) >= core_min_size);

    if (is_dense_cell)
    {
      if (is_boundary_point)
      {
        for (int jj = _mixed_offsets(k); jj < _mixed_offsets(k + 1); ++jj)
        {
          int j = _permute(jj);
          // No need to check core points here, all points in dense cells are
          // core
          if (distance(Access::get(_primitives, i),
                       Access::get(_primitives, j)) <= eps)
          {
            _union_find.merge_into(i, j);
            return ArborX::CallbackTreeTraversalControl::early_exit;
          }
        }
      }
      else
      {
        // TODO Not sure how important this is, as even without this check, the
        // for loop will terminate after the first iteration (or after a few
        // with i > j check)
        if (_point2offset(i) == k)
        {
          // The query point belongs to the same cell. Ignore, as the points
          // inside the dense box are processed separately
          return ArborX::CallbackTreeTraversalControl::normal_continuation;
        }

        for (int jj = _mixed_offsets(k); jj < _mixed_offsets(k + 1); ++jj)
        {
          int j = _permute(jj);
          // TODO: it is debatable whether the checks should be swapped. So
          // far, in some experiments (i > j) check first is faster.
          if (i > j && distance(Access::get(_primitives, i),
                                Access::get(_primitives, j)) <= eps)
          {
            _union_find.merge(i, j);
            // We connected to at least one point in the dense box, thus we
            // connected to all of them, so may terminate
            break;
          }
        }
      }
    }
    else
    {
      assert(_mixed_offsets(k + 1) - _mixed_offsets(k) == 1);

      int j = _permute(_mixed_offsets(k));

      // No need to check the distance here, as the fact that we are inside the
      // callback guarantees that it is <= eps
      if (_is_core_point(j))
      {
        if (is_boundary_point)
        {
          _union_find.merge_into(i, j);
          return ArborX::CallbackTreeTraversalControl::early_exit;
        }
        if (i > j)
        {
          _union_find.merge(i, j);
          return ArborX::CallbackTreeTraversalControl::normal_continuation;
        }
      }
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
      Kokkos::ViewAllocateWithoutInitializing("ArborX::dbscan::cell_indices"),
      n);
  Kokkos::parallel_for("ArborX::dbscan::computeCellIndices",
                       Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
                       KOKKOS_LAMBDA(int const index) {
                         auto const &xyz = Access::get(primitives, index);
                         cell_indices(index) = grid.cellIndex(xyz);
                       });
  return cell_indices;
}

template <typename ExecutionSpace, typename CellIndices>
Kokkos::View<int *, typename CellIndices::memory_space>
computeMixedOffsets(ExecutionSpace const &exec_space, int core_min_size,
                    CellIndices sorted_cell_indices, bool verbose = false)
{
  using MemorySpace = typename CellIndices::memory_space;
  int const n = sorted_cell_indices.size();

  int num_nonempty_cells;
  Kokkos::View<int *, MemorySpace> all_cell_offsets(
      Kokkos::ViewAllocateWithoutInitializing("ArborX::dbscan::cell_offsets"),
      n + 1);
  Kokkos::parallel_scan(
      "ArborX::dbscan::compute_cell_offsets",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n + 1),
      KOKKOS_LAMBDA(int i, int &update, bool final_pass) {
        bool const is_cell_first_index =
            (i == 0 || i == n ||
             sorted_cell_indices(i) != sorted_cell_indices(i - 1));
        if (is_cell_first_index)
        {
          if (final_pass)
            all_cell_offsets(update) = i;
          ++update;
        }
      },
      num_nonempty_cells);
  --num_nonempty_cells;
  Kokkos::resize(all_cell_offsets, num_nonempty_cells + 1);
  if (verbose)
    printf("#nonempty cells: %d\n", num_nonempty_cells);

  // NOTE: This is not ideal, as it does a thread may do a linear scan over a
  // range. The only saving grace is that this linear scan is guaranteed to not
  // exceed core_min_size. Would prefer a better way, but cannot figure out how.
  auto constexpr SWITCH_VALUE =
      std::numeric_limits<typename CellIndices::value_type>::max();
  auto modified_cell_indices = clone(exec_space, sorted_cell_indices);
  Kokkos::parallel_for(
      "ArborX::dbscan::modify_sorted_cell_indices",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_nonempty_cells),
      KOKKOS_LAMBDA(int i) {
        if (all_cell_offsets(i + 1) - all_cell_offsets(i) < core_min_size)
        {
          // The cell is not dense. Switch every other index to INT_MAX, so
          // that the next parallel_scan kernel will trigger on every point in
          // the cell.
          for (int j = all_cell_offsets(i) + 1; j < all_cell_offsets(i + 1);
               j += 2)
          {
            modified_cell_indices(j) = SWITCH_VALUE;
          }
        }
      });

  int num_mixed;
  Kokkos::View<int *, MemorySpace> mixed_offsets(
      Kokkos::ViewAllocateWithoutInitializing("ArborX::dbscan::mixed_offsets"),
      n + 1);
  Kokkos::parallel_scan(
      "ArborX::dbscan::compute_cell_index",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n + 1),
      KOKKOS_LAMBDA(int i, int &update, bool final_pass) {
        bool const is_cell_first_index =
            (i == 0 || i == n ||
             modified_cell_indices(i) != modified_cell_indices(i - 1));
        if (is_cell_first_index)
        {
          if (final_pass)
            mixed_offsets(update) = i;
          ++update;
        }
      },
      num_mixed);
  --num_mixed;
  Kokkos::resize(mixed_offsets, num_mixed + 1);
  if (verbose)
    printf("#mixed primitives: %d\n", num_mixed);

  return mixed_offsets;
}

template <typename ExecutionSpace, typename Labels, typename MixedOffsets,
          typename Permutation>
void unionFindWithinEachDenseCell(ExecutionSpace const &exec_space,
                                  int core_min_size, MixedOffsets mixed_offsets,
                                  Labels labels, Permutation permute,
                                  bool verbose = false)
{
  using MemorySpace = typename Permutation::memory_space;

  UnionFind<MemorySpace> union_find{labels};

  auto const num_nonempty_cells = mixed_offsets.size() - 1;
  auto const n = permute.size();

  Kokkos::View<int, MemorySpace> num_dense_cells(
      "ArborX::DBSCAN::num_dense_cells");
  Kokkos::View<int, MemorySpace> num_points_in_dense_cells(
      "ArborX::DBSCAN::num_points_in_dense_cells");

  Kokkos::parallel_for(
      "ArborX::dbscan::union_find_within_each_dense_box",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_nonempty_cells),
      KOKKOS_LAMBDA(int k) {
        int num_points_in_cell = mixed_offsets(k + 1) - mixed_offsets(k);
        if (num_points_in_cell >= core_min_size)
        {
          // Union all points in the dense box by going through pairs
          //
          // NOTE: This is pretty bad. A single thread will scan a linear range
          // corresponding to a dense cell. There is no upper limit on the
          // number of points in such a cell. In the pathological case, all
          // points may be contained in a single cell, making this completely
          // serial. Is there a way to do better?
          int i = permute(mixed_offsets(k));
          for (int jj = mixed_offsets(k) + 1; jj < mixed_offsets(k + 1); ++jj)
          {
            int j = permute(jj);
            union_find.merge(i, j);
          }

          Kokkos::atomic_fetch_add(&num_points_in_dense_cells(),
                                   num_points_in_cell);
          Kokkos::atomic_fetch_add(&num_dense_cells(), 1);
        }
      });

  if (verbose)
  {
    auto num_dense_cells_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, num_dense_cells);
    auto num_points_in_dense_cells_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, num_points_in_dense_cells);

    printf("#dense cells: %d\n", num_dense_cells_host());
    printf("#points in dense cells: %d [%.2f%%]\n",
           num_points_in_dense_cells_host(),
           (100.f * num_points_in_dense_cells_host()) / n);
  }
}

template <typename ExecutionSpace, typename Primitives, typename MixedOffsets,
          typename Permutation, typename NumNeighs, typename BVH>
void computeNumNeighbors(ExecutionSpace const &exec_space,
                         Primitives const &primitives, int core_min_size,
                         float eps, MixedOffsets mixed_offsets,
                         Permutation permute, NumNeighs num_neigh,
                         BVH const &bvh, bool verbose = false)
{
  using Access = AccessTraits<Primitives, PrimitivesTag>;
  using MemorySpace = typename Access::memory_space;

  auto const num_mixed = mixed_offsets.size() - 1;
  int const n = permute.size();

  Kokkos::View<decltype(attach(intersects(std::declval<Sphere>()),
                               std::declval<int>())) *,
               MemorySpace>
      sparse_predicates(Kokkos::ViewAllocateWithoutInitializing(
                            "ArborX::dbscan::sparse_predicates"),
                        n);
  Kokkos::View<int, MemorySpace> num_sparse_predicates(
      "ArborX::dbscan::num_sparse_predicates");
  Kokkos::parallel_for(
      "ArborX::dbscan::build_sparse_predicates_and_mark_dense_cores",
      Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_mixed),
      KOKKOS_LAMBDA(int k) {
        int num_points_in_cell = mixed_offsets(k + 1) - mixed_offsets(k);
        if (num_points_in_cell < core_min_size)
        {
          auto predicate_start = Kokkos::atomic_fetch_add(
              &num_sparse_predicates(), num_points_in_cell);
          for (int jj = mixed_offsets(k); jj < mixed_offsets(k + 1); ++jj)
          {
            int j = permute(jj);
            sparse_predicates(predicate_start++) = attach(
                intersects(Sphere{Access::get(primitives, j), eps}), (int)j);
          }
        }
        else
        {
          // Mark all points inside dense cells as core
          //
          // NOTE: Again a bad serial scan, same issue as in
          // union_find_within_each_dense_cell.
          for (int jj = mixed_offsets(k); jj < mixed_offsets(k + 1); ++jj)
            num_neigh(permute(jj)) = INT_MAX;
        }
      });
  auto num_sparse_predicates_host = Kokkos::create_mirror_view_and_copy(
      Kokkos::HostSpace{}, num_sparse_predicates);
  Kokkos::resize(sparse_predicates, num_sparse_predicates_host());

  if (verbose)
    printf("#sparse predicates: %d\n", (int)num_sparse_predicates_host());

  // We only need to determine core point through search in non-dense cells
  bvh.query(
      exec_space, sparse_predicates,
      FDBSCANDenseBoxCorePointsCallback<
          MemorySpace, Primitives, decltype(mixed_offsets), decltype(permute)>{
          num_neigh, primitives, mixed_offsets, permute, core_min_size, eps});
}

} // namespace Details
} // namespace ArborX

#endif
