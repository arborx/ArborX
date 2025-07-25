/****************************************************************************
 * Copyright (c) 2025, ArborX authors                                       *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_DBSCAN_HPP
#define ARBORX_DBSCAN_HPP

#include <ArborX_Box.hpp>
#include <ArborX_LinearBVH.hpp>
#include <ArborX_Sphere.hpp>
#include <detail/ArborX_AccessTraits.hpp>
#include <detail/ArborX_CartesianGrid.hpp>
#include <detail/ArborX_FDBSCAN.hpp>
#include <detail/ArborX_FDBSCANDenseBox.hpp>
#include <detail/ArborX_HalfTraversal.hpp>
#include <detail/ArborX_PredicateHelpers.hpp>
#include <kokkos_ext/ArborX_KokkosExtAccessibilityTraits.hpp>
#include <kokkos_ext/ArborX_KokkosExtStdAlgorithms.hpp>
#include <misc/ArborX_SortUtils.hpp>
#include <misc/ArborX_Utils.hpp> // sortObjects

namespace ArborX
{

namespace Details
{

// All points are marked as if they were core points minpts = 2 case.
// Obviously, this is not true. However, in the algorithms it is used only for
// pairs of points within the distance eps, in which case it is correct.
struct CCSCorePoints
{
  KOKKOS_FUNCTION bool operator()(int) const { return true; }
};

template <typename MemorySpace>
struct DBSCANCorePoints
{
  Kokkos::View<int *, MemorySpace> _num_neigh;
  int _core_min_size;

  KOKKOS_FUNCTION bool operator()(int const i) const
  {
    return _num_neigh(i) >= _core_min_size;
  }
};

template <typename Coordinate>
struct WithinRadiusGetter
{
  Coordinate _r;

  template <typename Point, typename Index>
  KOKKOS_FUNCTION auto
  operator()(PairValueIndex<Point, Index> const &pair) const
  {
    static_assert(GeometryTraits::is_point_v<Point>);

    constexpr int DIM = GeometryTraits::dimension_v<Point>;
    using ArborX::intersects;
    return intersects(
        Sphere{convert<::ArborX::Point<DIM, Coordinate>>(pair.value), _r});
  }
};

template <typename Points, typename PermuteFilter>
struct PointsWithRadiusReorderedAndFiltered
{
  Points _points;
  GeometryTraits::coordinate_type_t<typename Points::value_type> _r;
  PermuteFilter _filter;
};

// Mixed primitives consist of a set of boxes corresponding to dense cells,
// followed by boxes corresponding to points in non-dense cells.
template <typename Points, typename DenseCellOffsets, typename CellIndices,
          typename Permutation>
struct MixedBoxPrimitives
{
  using Point = typename Points::value_type;
  Points _points;
  CartesianGrid<GeometryTraits::dimension_v<Point>,
                GeometryTraits::coordinate_type_t<Point>>
      _grid;
  DenseCellOffsets _dense_cell_offsets;
  int _num_points_in_dense_cells; // to avoid lastElement() in AccessTraits
  CellIndices _sorted_cell_indices;
  Permutation _permute;
};

} // namespace Details

template <typename Primitives, typename PermuteFilter>
struct AccessTraits<
    Details::PointsWithRadiusReorderedAndFiltered<Primitives, PermuteFilter>>
{
  using memory_space = typename Primitives::memory_space;
  using Predicates =
      Details::PointsWithRadiusReorderedAndFiltered<Primitives, PermuteFilter>;

  static KOKKOS_FUNCTION size_t size(Predicates const &w)
  {
    return w._filter.extent(0);
  }
  static KOKKOS_FUNCTION auto get(Predicates const &w, size_t i)
  {
    int index = w._filter(i);
    auto const &point = w._points(index);
    using Point = std::decay_t<decltype(point)>;
    constexpr int DIM = GeometryTraits::dimension_v<Point>;
    using Coordinate = GeometryTraits::coordinate_type_t<Point>;
    // FIXME reinterpret_cast is dangerous here if access traits return user
    // point structure (e.g., struct MyPoint { float y; float x; })
    auto const &hyper_point =
        reinterpret_cast<::ArborX::Point<DIM, Coordinate> const &>(point);
    return attach(intersects(Sphere{hyper_point, w._r}), (int)index);
  }
};

template <typename Points, typename MixedOffsets, typename CellIndices,
          typename Permutation>
struct AccessTraits<
    Details::MixedBoxPrimitives<Points, MixedOffsets, CellIndices, Permutation>>
{
  using Primitives = Details::MixedBoxPrimitives<Points, MixedOffsets,
                                                 CellIndices, Permutation>;
  static KOKKOS_FUNCTION std::size_t size(Primitives const &w)
  {
    auto const &dco = w._dense_cell_offsets;

    auto const n = w._permute.size();
    auto num_dense_primitives = dco.size() - 1;
    auto num_sparse_primitives = n - w._num_points_in_dense_cells;

    return num_dense_primitives + num_sparse_primitives;
  }
  static KOKKOS_FUNCTION auto get(Primitives const &w, std::size_t i)
  {
    auto const &dco = w._dense_cell_offsets;

    auto num_dense_primitives = dco.size() - 1;
    if (i < num_dense_primitives)
    {
      // For a primitive corresponding to a dense cell, use that cell's box.
      // It may not be tight around the points inside, but is cheap to
      // compute.
      auto cell_index = w._sorted_cell_indices(dco(i));
      return w._grid.cellBox(cell_index);
    }

    // For a primitive corresponding to a point in a non-dense cell, use that
    // point. But first, figure out its index, which requires some
    // computations.
    i = (i - num_dense_primitives) + w._num_points_in_dense_cells;

    auto const &point = w._points(w._permute(i));

    using Point = std::decay_t<decltype(point)>;
    constexpr int DIM = GeometryTraits::dimension_v<Point>;
    using Coordinate = typename GeometryTraits::coordinate_type<Point>::type;

    // FIXME reinterpret_cast is dangerous here if access traits return user
    // point structure (e.g., struct MyPoint { float y; float x; })
    auto const &hyper_point =
        reinterpret_cast<::ArborX::Point<DIM, Coordinate> const &>(point);
    return Box{hyper_point, hyper_point};
  }
  using memory_space = typename MixedOffsets::memory_space;
};

namespace DBSCAN
{

enum class Implementation
{
  FDBSCAN,
  FDBSCAN_DenseBox
};

struct Parameters
{
  // Print timers to standard output
  bool _verbose = false;
  // Algorithm implementation (FDBSCAN or FDBSCAN-DenseBox)
  Implementation _implementation = Implementation::FDBSCAN_DenseBox;

  Parameters &setVerbosity(bool verbose)
  {
    _verbose = verbose;
    return *this;
  }
  Parameters &setImplementation(Implementation impl)
  {
    _implementation = impl;
    return *this;
  }
};
} // namespace DBSCAN

template <typename ExecutionSpace, typename Primitives, typename Coordinate>
Kokkos::View<int *, typename AccessTraits<Primitives>::memory_space>
dbscan(ExecutionSpace const &exec_space, Primitives const &primitives,
       Coordinate eps, int core_min_size,
       DBSCAN::Parameters const &parameters = DBSCAN::Parameters())
{
  Kokkos::Profiling::pushRegion("ArborX::DBSCAN");

  namespace KokkosExt = ArborX::Details::KokkosExt;

  using Points = Details::AccessValues<Primitives>;
  using MemorySpace = typename Points::memory_space;

  static_assert(
      KokkosExt::is_accessible_from<MemorySpace, ExecutionSpace>::value,
      "Primitives must be accessible from the execution space");

  ARBORX_ASSERT(eps > 0);
  ARBORX_ASSERT(core_min_size >= 2);

#ifdef KOKKOS_ENABLE_SERIAL
  using UnionFind = Details::UnionFind<
      MemorySpace,
      /*DoSerial=*/std::is_same_v<ExecutionSpace, Kokkos::Serial>>;
#else
  using UnionFind = Details::UnionFind<MemorySpace>;
#endif

  using Point = typename Points::value_type;
  static_assert(GeometryTraits::is_point_v<Point>);
  constexpr int DIM = GeometryTraits::dimension_v<Point>;
  static_assert(
      std::is_same_v<typename GeometryTraits::coordinate_type<Point>::type,
                     Coordinate>);

  using Box = Box<DIM, Coordinate>;

  bool const is_special_case = (core_min_size == 2);

  bool const verbose = parameters._verbose;

  Points points{primitives}; // NOLINT
  int const n = points.size();

  Kokkos::View<int *, MemorySpace> num_neigh("ArborX::DBSCAN::num_neighbors",
                                             0);

  Kokkos::View<int *, MemorySpace> labels("ArborX::DBSCAN::labels", 0);

  if (parameters._implementation == DBSCAN::Implementation::FDBSCAN)
  {
    // Build the tree
    Kokkos::Profiling::pushRegion("ArborX::DBSCAN::tree_construction");
    BoundingVolumeHierarchy bvh(exec_space,
                                Experimental::attach_indices(points));
    Kokkos::Profiling::popRegion();

    // Initialize labels after the hierarchy construction to lower memory high
    // water mark
    Kokkos::resize(Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing),
                   labels, n);
    KokkosExt::iota(exec_space, labels);

    Kokkos::Profiling::pushRegion("ArborX::DBSCAN::clusters");
    if (is_special_case)
    {
      // Perform the queries and build clusters through callback
      using CorePoints = Details::CCSCorePoints;
      Kokkos::Profiling::pushRegion("ArborX::DBSCAN::clusters::query");
      Details::HalfTraversal(
          exec_space, bvh,
          Details::FDBSCANCallback<UnionFind, CorePoints>{labels, CorePoints{}},
          Details::WithinRadiusGetter<Coordinate>{eps});
      Kokkos::Profiling::popRegion();
    }
    else
    {
      auto const predicates = ArborX::Experimental::attach_indices(
          ArborX::Experimental::make_intersects(points, eps));

      // Determine core points
      Kokkos::Profiling::pushRegion("ArborX::DBSCAN::clusters::num_neigh");
      Kokkos::resize(Kokkos::view_alloc(exec_space), num_neigh, n);
      bvh.query(exec_space, predicates,
                Details::CountUpToN<MemorySpace>{num_neigh, core_min_size});
      Kokkos::Profiling::popRegion();

      using CorePoints = Details::DBSCANCorePoints<MemorySpace>;

      // Perform the queries and build clusters through callback
      Kokkos::Profiling::pushRegion("ArborX::DBSCAN::clusters::query");
      Details::HalfTraversal(exec_space, bvh,
                             Details::FDBSCANCallback<UnionFind, CorePoints>{
                                 labels, CorePoints{num_neigh, core_min_size}},
                             Details::WithinRadiusGetter<Coordinate>{eps});
      Kokkos::Profiling::popRegion();
    }
  }
  else if (parameters._implementation ==
           DBSCAN::Implementation::FDBSCAN_DenseBox)
  {
    // Find dense boxes
    Kokkos::Profiling::pushRegion("ArborX::DBSCAN::dense_cells");
    Box bounds;
    Details::TreeConstruction::calculateBoundingBoxOfTheScene(
        exec_space,
        Details::Indexables{points, Experimental::DefaultIndexableGetter{}},
        bounds);

    // The cell length is chosen to be eps/sqrt(dimension), so that any two
    // points within the same cell are within eps distance of each other.
    auto const h = eps / Kokkos::sqrt((Coordinate)DIM);
    Details::CartesianGrid const grid(bounds, h);

    auto cell_indices = Details::computeCellIndices(exec_space, points, grid);

    auto permute = Details::sortObjects(exec_space, cell_indices);
    auto &sorted_cell_indices = cell_indices; // alias

    int num_nonempty_cells;
    int num_points_in_dense_cells;
    {
      // Reorder indices and permutation so that the dense cells go first
      Kokkos::View<int *, MemorySpace> cell_offsets(
          "ArborX::DBSCAN::cell_offsets", 0);
      Details::computeOffsetsInOrderedView(exec_space, sorted_cell_indices,
                                           cell_offsets);
      num_nonempty_cells = cell_offsets.size() - 1;

      num_points_in_dense_cells = Details::reorderDenseAndSparseCells(
          exec_space, cell_offsets, core_min_size, sorted_cell_indices,
          permute);
    }
    int num_points_in_sparse_cells = n - num_points_in_dense_cells;

    auto dense_sorted_cell_indices = Kokkos::subview(
        sorted_cell_indices, Kokkos::make_pair(0, num_points_in_dense_cells));

    Kokkos::View<int *, MemorySpace> dense_cell_offsets(
        "ArborX::DBSCAN::dense_cell_offsets", 0);
    Details::computeOffsetsInOrderedView(exec_space, dense_sorted_cell_indices,
                                         dense_cell_offsets);
    int num_dense_cells = dense_cell_offsets.size() - 1;
    if (verbose)
    {
      printf("h = %e, n = [%zu", h, grid.extent(0));
      for (int d = 1; d < DIM; ++d)
        printf(", %zu", grid.extent(d));
      printf("]\n");
      printf("#nonempty cells     : %10d\n", num_nonempty_cells);
      printf("#dense cells        : %10d [%.2f%%]\n", num_dense_cells,
             (100.f * num_dense_cells) / num_nonempty_cells);
      printf("#dense cell points  : %10d [%.2f%%]\n", num_points_in_dense_cells,
             (100.f * num_points_in_dense_cells) / n);
      printf("#mixed primitives   : %10d\n",
             num_dense_cells + num_points_in_sparse_cells);
    }

    Kokkos::resize(Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing),
                   labels, n);
    KokkosExt::iota(exec_space, labels);

    Details::unionFindWithinEachDenseCell(exec_space, dense_sorted_cell_indices,
                                          permute, UnionFind{labels});

    Kokkos::Profiling::popRegion();

    // Build the tree
    Kokkos::Profiling::pushRegion("ArborX::DBSCAN::tree_construction");
    Details::MixedBoxPrimitives<Points, decltype(dense_cell_offsets),
                                decltype(cell_indices), decltype(permute)>
        mixed_primitives{points,
                         grid,
                         dense_cell_offsets,
                         num_points_in_dense_cells,
                         sorted_cell_indices,
                         permute};

    BoundingVolumeHierarchy bvh(exec_space,
                                Experimental::attach_indices(mixed_primitives));

    Kokkos::Profiling::popRegion();

    Kokkos::Profiling::pushRegion("ArborX::DBSCAN::clusters");

    if (is_special_case)
    {
      // Perform the queries and build clusters through callback
      using CorePoints = Details::CCSCorePoints;
      Kokkos::Profiling::pushRegion("ArborX::DBSCAN::clusters::query");
      auto const predicates = Experimental::attach_indices(
          Experimental::make_intersects(points, eps));
      bvh.query(exec_space, predicates,
                Details::FDBSCANDenseBoxCallback<UnionFind, CorePoints, Points,
                                                 decltype(dense_cell_offsets),
                                                 decltype(permute)>{
                    labels, CorePoints{}, points, dense_cell_offsets,
                    exec_space, permute, eps});
      Kokkos::Profiling::popRegion();
    }
    else
    {
      // Determine core points
      Kokkos::Profiling::pushRegion("ArborX::DBSCAN::clusters::num_neigh");
      Kokkos::resize(Kokkos::view_alloc(exec_space), num_neigh, n);
      // Set num neighbors for points in dense cells to max, so that they are
      // automatically core points
      Kokkos::parallel_for(
          "ArborX::DBSCAN::mark_dense_cells_core_points",
          Kokkos::RangePolicy(exec_space, 0, num_points_in_dense_cells),
          KOKKOS_LAMBDA(int i) { num_neigh(permute(i)) = INT_MAX; });
      // Count neighbors for points in sparse cells
      auto sparse_permute = Kokkos::subview(
          permute, Kokkos::make_pair(num_points_in_dense_cells, n));

      auto const sparse_predicates =
          Details::PointsWithRadiusReorderedAndFiltered<
              Points, decltype(sparse_permute)>{points, eps, sparse_permute};
      bvh.query(exec_space, sparse_predicates,
                Details::CountUpToN_DenseBox<MemorySpace, Points,
                                             decltype(dense_cell_offsets),
                                             decltype(permute)>(
                    num_neigh, points, dense_cell_offsets, permute,
                    core_min_size, eps, core_min_size));
      Kokkos::Profiling::popRegion();

      using CorePoints = Details::DBSCANCorePoints<MemorySpace>;

      // Perform the queries and build clusters through callback
      Kokkos::Profiling::pushRegion("ArborX::DBSCAN::clusters::query");
      auto const predicates = Experimental::attach_indices(
          Experimental::make_intersects(points, eps));
      bvh.query(exec_space, predicates,
                Details::FDBSCANDenseBoxCallback<UnionFind, CorePoints, Points,
                                                 decltype(dense_cell_offsets),
                                                 decltype(permute)>{
                    labels, CorePoints{num_neigh, core_min_size}, points,
                    dense_cell_offsets, exec_space, permute, eps});
      Kokkos::Profiling::popRegion();
    }
  }

  // Per [1]:
  //
  // ```
  // The finalization kernel will, ultimately, make all parents
  // point directly to the representative.
  // ```
  Kokkos::View<int *, MemorySpace> cluster_sizes(
      Kokkos::view_alloc(exec_space, "ArborX::DBSCAN::cluster_sizes"), n);
  Kokkos::parallel_for(
      "ArborX::DBSCAN::finalize_labels", Kokkos::RangePolicy(exec_space, 0, n),
      KOKKOS_LAMBDA(int const i) {
        // ##### ECL license (see LICENSE.ECL) #####
        int next;
        int vstat = labels(i);
        int const old = vstat;
        while (vstat > (next = labels(vstat)))
        {
          vstat = next;
        }
        if (vstat != old)
          labels(i) = vstat;

        Kokkos::atomic_inc(&cluster_sizes(labels(i)));
      });

  // Mark noise points
  // Account for the special case where a core point is connected to only
  // border points that got assigned to other clusters. This will result in a
  // cluster consisting of a single core point.
  Details::DBSCANCorePoints<MemorySpace> is_core{num_neigh, core_min_size};
  Kokkos::parallel_for(
      "ArborX::DBSCAN::mark_noise", Kokkos::RangePolicy(exec_space, 0, n),
      KOKKOS_LAMBDA(int const i) {
        if (cluster_sizes(labels(i)) == 1 && (is_special_case || !is_core(i)))
          labels(i) = -1;
      });
  Kokkos::Profiling::popRegion();

  Kokkos::Profiling::popRegion();

  return labels;
}

} // namespace ArborX

#endif
