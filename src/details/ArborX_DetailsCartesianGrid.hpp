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
  size_t cellIndex(Point const &point) const
  {
    auto const &min_corner = _bounds.minCorner();
    size_t i = std::floor((point[0] - min_corner[0]) / _hx);
    size_t j = std::floor((point[1] - min_corner[1]) / _hy);
    size_t k = std::floor((point[2] - min_corner[2]) / _hz);
    return k * _nx * _ny + j * _nx + i;
  }

  KOKKOS_FUNCTION
  Box cellBox(size_t cell_index) const
  {
    auto const &min_corner = _bounds.minCorner();

    auto i = cell_index % _nx;
    auto j = (cell_index / _nx) % _ny;
    auto k = cell_index / (_nx * _ny);
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

} // namespace Details
} // namespace ArborX

#endif
