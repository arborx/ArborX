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

#ifndef ARBORX_GRID_HPP
#define ARBORX_GRID_HPP

#include <ArborX_AccessTraits.hpp>
#include <ArborX_Box.hpp>
#include <ArborX_CrsGraphWrapper.hpp>
#include <ArborX_DetailsCartesianGrid.hpp>
#include <ArborX_DetailsGridImpl.hpp>
#include <ArborX_DetailsKokkosExtAccessibilityTraits.hpp>

#include <Kokkos_Core.hpp>

namespace ArborX
{

template <typename MemorySpace>
class Grid
{
public:
  using memory_space = MemorySpace;
  static_assert(Kokkos::is_memory_space<MemorySpace>::value, "");
  using size_type = typename MemorySpace::size_type;

  Grid() = default;

  template <typename ExecutionSpace, typename Primitives>
  Grid(ExecutionSpace const &exec_space, Primitives const &primitives, float hx,
       float hy, float hz);

  KOKKOS_FUNCTION
  size_type size() const noexcept { return _size; }

  KOKKOS_FUNCTION
  bool empty() const noexcept { return size() == 0; }

  KOKKOS_FUNCTION
  Box bounds() const noexcept { return _grid._bounds; }

  template <typename ExecutionSpace, typename Predicates, typename Callback,
            typename Ignore = int>
  void query(ExecutionSpace const &exec_space, Predicates const &predicates,
             Callback const &callback, Ignore = Ignore()) const;

private:
  size_type _size;
  Kokkos::View<Point *, MemorySpace> _points;
  Details::CartesianGrid _grid;
  Kokkos::View<size_t *, MemorySpace> _bin_indices_1d;
  Kokkos::View<int ***, MemorySpace> _bin_offsets_3d;
  Kokkos::View<int ***, MemorySpace> _bin_counts_3d;
  Kokkos::View<unsigned *, MemorySpace> _permute;
};

template <typename MemorySpace>
template <typename ExecutionSpace, typename Primitives>
Grid<MemorySpace>::Grid(ExecutionSpace const &exec_space,
                        Primitives const &primitives, float hx, float hy,
                        float hz)
    : _size(AccessTraits<Primitives, PrimitivesTag>::size(primitives))
    , _points(Kokkos::view_alloc(Kokkos::WithoutInitializing,
                                 "ArborX::Grid::points"),
              _size)
    , _grid()
    , _bin_indices_1d("ArborX::Grid::bin_indices_1d", 0)
    , _bin_offsets_3d("ArborX::Grid::bin_offsets_3d", 0, 0, 0)
    , _bin_counts_3d("ArborX::Grid::bin_counts_3d", 0, 0, 0)
{
  static_assert(
      KokkosExt::is_accessible_from<MemorySpace, ExecutionSpace>::value, "");
  Details::check_valid_access_traits(PrimitivesTag{}, primitives);
  using Access = AccessTraits<Primitives, PrimitivesTag>;
  static_assert(KokkosExt::is_accessible_from<typename Access::memory_space,
                                              ExecutionSpace>::value,
                "Primitives must be accessible from the execution exec_space");

  static_assert(
      std::is_same<typename Details::AccessTraitsHelper<Access>::type, Point>{},
      "AccessTraits<Primitives,PrimitivesTag>::get() return type "
      "must decay to Point");

  Kokkos::Profiling::pushRegion("ArborX::Grid::Grid");

  Box bounds;
  Details::GridImpl::calculateBoundingBoxOfTheScene(exec_space, primitives,
                                                    bounds);

  _grid = Details::CartesianGrid(bounds, hx, hy, hz);

  auto indices = Details::computeCellIndices(exec_space, primitives, _grid);

  _permute = Details::sortObjects(exec_space, indices);
  auto &sorted_indices = indices; // alias

  Details::GridImpl::initializePoints(exec_space, primitives, _permute,
                                      _points);

  Kokkos::View<int *, MemorySpace> bin_offsets_1d(
      "ArborX::Grid::bin_offsets_1d", 0);
  Details::computeOffsetsInOrderedView(exec_space, sorted_indices,
                                       bin_offsets_1d);
  auto const num_bins = bin_offsets_1d.extent(0) - 1;

  std::cout << "num_bins = " << num_bins << std::endl;

  Details::GridImpl::computeBinIndices(exec_space, bin_offsets_1d,
                                       sorted_indices, _bin_indices_1d);

  Details::GridImpl::convertBinOffsetsTo3D(exec_space, _grid, bin_offsets_1d,
                                           _bin_indices_1d, _bin_offsets_3d,
                                           _bin_counts_3d);

  Kokkos::Profiling::popRegion();
}

template <typename MemorySpace>
template <typename ExecutionSpace, typename Predicates, typename Callback,
          typename Ignore>
void Grid<MemorySpace>::query(ExecutionSpace const &exec_space,
                              Predicates const &predicates,
                              Callback const &callback, Ignore) const
{
  static_assert(
      KokkosExt::is_accessible_from<MemorySpace, ExecutionSpace>::value, "");
  Details::check_valid_access_traits(PredicatesTag{}, predicates);
  using Access = AccessTraits<Predicates, PredicatesTag>;
  static_assert(KokkosExt::is_accessible_from<typename Access::memory_space,
                                              ExecutionSpace>::value,
                "Predicates must be accessible from the execution exec_space");
  using Tag = typename Details::AccessTraitsHelper<Access>::tag;
  static_assert(std::is_same<Tag, Details::SpatialPredicateTag>{},
                "nearest query not implemented yet");
  Details::check_valid_callback(callback, predicates);

  Kokkos::Profiling::pushRegion("ArborX::Grid::query::spatial");

  Details::GridImpl::query(exec_space, predicates, callback, _permute, _points,
                           _grid, _bin_indices_1d, _bin_offsets_3d,
                           _bin_counts_3d);

  Kokkos::Profiling::popRegion();
}

} // namespace ArborX

#endif
