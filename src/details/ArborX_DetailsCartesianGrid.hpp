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

#ifndef ARBORX_DETAILS_CARTESIAN_GRID_HPP
#define ARBORX_DETAILS_CARTESIAN_GRID_HPP

#include <ArborX_Exception.hpp>
#include <ArborX_GeometryTraits.hpp>
#include <ArborX_HyperBox.hpp>

#include <Kokkos_Macros.hpp>
#include <Kokkos_MathematicalFunctions.hpp> // floor

#include <cassert>

namespace ArborX::Details
{

template <int DIM>
struct CartesianGrid
{
private:
  using Box = ExperimentalHyperGeometry::Box<DIM>;

public:
  static constexpr int dim = DIM;

  CartesianGrid(Box const &bounds, float h)
      : _bounds(bounds)
  {
    ARBORX_ASSERT(h > 0);
    for (int d = 0; d < DIM; ++d)
      _h[d] = h;
    buildGrid();
  }
  CartesianGrid(Box const &bounds, float const h[DIM])
      : _bounds(bounds)
  {
    for (int d = 0; d < DIM; ++d)
    {
      ARBORX_ASSERT(_h[d] > 0);
      _h[d] = h[d];
    }
    buildGrid();
  }

  template <typename Point, typename Enable = std::enable_if_t<
                                GeometryTraits::is_point<Point>{}>>
  KOKKOS_FUNCTION size_t cellIndex(Point const &point) const
  {
    static_assert(GeometryTraits::dimension_v<Point> == DIM);

    auto const &min_corner = _bounds.minCorner();
    size_t s = 0;
    for (int d = DIM - 1; d >= 0; --d)
    {
      int i = Kokkos::floor((point[d] - min_corner[d]) / _h[d]);
      s = s * _n[d] + i;
    }
    return s;
  }

  KOKKOS_FUNCTION
  Box cellBox(size_t cell_index) const
  {
    auto min = _bounds.minCorner();
    decltype(min) max;

    // This code may suffer from loss of precision depending on the problem
    // bounds and h. We try to detect this case in the constructor.
    for (int d = 0; d < DIM; ++d)
    {
      auto i = cell_index % _n[d];
      cell_index /= _n[d];

      max[d] = min[d] + (i + 1) * _h[d];
      min[d] += i * _h[d];
    }
    return {min, max};
  }

  KOKKOS_FUNCTION
  auto extent(int d) const
  {
    assert(0 <= d && d < DIM);
    return _n[d];
  }

private:
  void buildGrid()
  {
    auto const &min_corner = _bounds.minCorner();
    auto const &max_corner = _bounds.maxCorner();
    for (int d = 0; d < DIM; ++d)
    {
      auto delta = max_corner[d] - min_corner[d];
      if (delta != 0)
      {
        _n[d] = std::ceil(delta / _h[d]);
        ARBORX_ASSERT(_n[d] > 0);
      }
      else
      {
        _n[d] = 1;
      }
    }

    // Catch potential overflow in grid cell indices early. This is a
    // conservative check as an actual overflow may not occur, depending on
    // which cells are filled.
    constexpr auto max_size_t = std::numeric_limits<size_t>::max();
    auto m = max_size_t;
    for (int d = 1; d < DIM; ++d)
    {
      m /= _n[d - 1];
      ARBORX_ASSERT(_n[d] < m);
    }

    // Catch a potential loss of precision that may happen in cellBox() and can
    // lead to wrong results.
    //
    // The machine precision by itself is not sufficient. In some experiments
    // run with a full NGSIM datasets, values below 3 could still produce wrong
    // results. This may still not be conservative enough, but all runs passed
    // verification when this warning was not triggered.
    constexpr auto eps = 5 * std::numeric_limits<float>::epsilon();
    for (int d = 0; d < DIM; ++d)
    {
      if (std::abs(_h[d] / min_corner[d]) < eps)
        throw std::runtime_error(
            "ArborX exception: FDBSCAN-DenseBox algorithm will experience loss "
            "of precision, undetectably producing wrong results. Please switch "
            "to using FDBSCAN.");
    }
  }

  Box _bounds;
  float _h[DIM];
  size_t _n[DIM];
};

} // namespace ArborX::Details

#endif
