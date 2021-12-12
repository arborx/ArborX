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

#ifndef ARBORX_DETAILSCARTESIANGRID_HPP
#define ARBORX_DETAILSCARTESIANGRID_HPP

#include <ArborX_AccessTraits.hpp>
#include <ArborX_Box.hpp>
#include <ArborX_Exception.hpp>

#include <Kokkos_Macros.hpp>

#include <cassert>

namespace ArborX
{
namespace Details
{

struct CartesianGrid
{
  Box _bounds;
  float _hx;
  float _hy;
  float _hz;
  size_t _nx;
  size_t _ny;
  size_t _nz;

  CartesianGrid() {}

  CartesianGrid(Box const &bounds, float h)
      : _bounds(bounds)
      , _hx(h)
      , _hy(h)
      , _hz(h)
  {
    buildGrid();
  }
  CartesianGrid(Box const &bounds, float hx, float hy, float hz)
      : _bounds(bounds)
      , _hx(hx)
      , _hy(hy)
      , _hz(hz)
  {
    buildGrid();
  }

  KOKKOS_FUNCTION
  void cellIndex2Triplet(size_t index, size_t &i, size_t &j, size_t &k) const
  {
    i = index % _nx;
    j = (index / _nx) % _ny;
    k = index / (_nx * _ny);
  }

  KOKKOS_FUNCTION
  size_t triplet2CellIndex(size_t i, size_t j, size_t k) const
  {
    return k * _nx * _ny + j * _nx + i;
  }

  KOKKOS_FUNCTION
  size_t cellIndex(Point const &point) const
  {
    auto const &min_corner = _bounds.minCorner();
    size_t i = std::floor((point[0] - min_corner[0]) / _hx);
    size_t j = std::floor((point[1] - min_corner[1]) / _hy);
    size_t k = std::floor((point[2] - min_corner[2]) / _hz);
    return triplet2CellIndex(i, j, k);
  }

  KOKKOS_FUNCTION
  Box cellBox(size_t index) const
  {
    auto const &min_corner = _bounds.minCorner();

    size_t i;
    size_t j;
    size_t k;
    cellIndex2Triplet(index, i, j, k);
    return {{min_corner[0] + i * _hx, min_corner[1] + j * _hy,
             min_corner[2] + k * _hz},
            {min_corner[0] + (i + 1) * _hx, min_corner[1] + (j + 1) * _hy,
             min_corner[2] + (k + 1) * _hz}};
  }

private:
  void buildGrid()
  {
    auto const &min_corner = _bounds.minCorner();
    auto const &max_corner = _bounds.maxCorner();
    _nx = std::ceil((max_corner[0] - min_corner[0]) / _hx);
    _ny = std::ceil((max_corner[1] - min_corner[1]) / _hy);
    _nz = std::ceil((max_corner[2] - min_corner[2]) / _hz);

    // Catch potential overflow in grid cell indices early. This is a
    // conservative check as an actual overflow may not occur, depending on
    // which cells are filled.
    constexpr auto max_size_t = std::numeric_limits<size_t>::max();
    ARBORX_ASSERT(_nx == 0 || _ny == 0 || _nz == 0 ||
                  (_ny < max_size_t / _nx && _nz < max_size_t / (_nx * _ny)));
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
      Kokkos::view_alloc(Kokkos::WithoutInitializing,
                         "ArborX::Grid::cell_indices"),
      n);
  Kokkos::parallel_for("ArborX::Grid::compute_cell_indices",
                       Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, n),
                       KOKKOS_LAMBDA(int i) {
                         Point const &xyz = Access::get(primitives, i);
                         cell_indices(i) = grid.cellIndex(xyz);
                       });
  return cell_indices;
}

} // namespace Details
} // namespace ArborX

#endif
