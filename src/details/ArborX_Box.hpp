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

#ifndef ARBORX_BOX_HPP
#define ARBORX_BOX_HPP

#include <ArborX_DetailsKokkosExtArithmeticTraits.hpp>
#include <ArborX_DetailsKokkosExtMinMaxOperations.hpp>
#include <ArborX_Point.hpp>

#include <Kokkos_Macros.hpp>

namespace ArborX
{
/**
 * Axis-Aligned Bounding Box. This is just a thin wrapper around an array of
 * size 2x spatial dimension with a default constructor to initialize
 * properly an "empty" box.
 */
struct Box
{
  KOKKOS_DEFAULTED_FUNCTION
  constexpr Box() = default;

  KOKKOS_INLINE_FUNCTION
  constexpr Box(Point const &min_corner, Point const &max_corner)
      : _min_corner(min_corner)
      , _max_corner(max_corner)
  {
  }

  KOKKOS_INLINE_FUNCTION
  constexpr Point &minCorner() { return _min_corner; }

  KOKKOS_INLINE_FUNCTION
  constexpr Point const &minCorner() const { return _min_corner; }

  KOKKOS_INLINE_FUNCTION
  Point volatile &minCorner() volatile { return _min_corner; }

  KOKKOS_INLINE_FUNCTION
  Point volatile const &minCorner() volatile const { return _min_corner; }

  KOKKOS_INLINE_FUNCTION
  constexpr Point &maxCorner() { return _max_corner; }

  KOKKOS_INLINE_FUNCTION
  constexpr Point const &maxCorner() const { return _max_corner; }

  KOKKOS_INLINE_FUNCTION
  Point volatile &maxCorner() volatile { return _max_corner; }

  KOKKOS_INLINE_FUNCTION
  Point volatile const &maxCorner() volatile const { return _max_corner; }

  Point _min_corner = {{KokkosExt::ArithmeticTraits::finite_max<float>::value,
                        KokkosExt::ArithmeticTraits::finite_max<float>::value,
                        KokkosExt::ArithmeticTraits::finite_max<float>::value}};
  Point _max_corner = {{KokkosExt::ArithmeticTraits::finite_min<float>::value,
                        KokkosExt::ArithmeticTraits::finite_min<float>::value,
                        KokkosExt::ArithmeticTraits::finite_min<float>::value}};

  KOKKOS_FUNCTION Box &operator+=(Box const &other)
  {
    using KokkosExt::max;
    using KokkosExt::min;

    for (int d = 0; d < 3; ++d)
    {
      minCorner()[d] = min(minCorner()[d], other.minCorner()[d]);
      maxCorner()[d] = max(maxCorner()[d], other.maxCorner()[d]);
    }
    return *this;
  }

  KOKKOS_FUNCTION void operator+=(Box const volatile &other) volatile
  {
    using KokkosExt::max;
    using KokkosExt::min;

    for (int d = 0; d < 3; ++d)
    {
      minCorner()[d] = min(minCorner()[d], other.minCorner()[d]);
      maxCorner()[d] = max(maxCorner()[d], other.maxCorner()[d]);
    }
  }

  KOKKOS_FUNCTION Box &operator+=(Point const &point)
  {
    using KokkosExt::max;
    using KokkosExt::min;

    for (int d = 0; d < 3; ++d)
    {
      minCorner()[d] = min(minCorner()[d], point[d]);
      maxCorner()[d] = max(maxCorner()[d], point[d]);
    }
    return *this;
  }

// FIXME Temporary workaround until we clarify requirements on the Kokkos side.
#if defined(KOKKOS_ENABLE_OPENMPTARGET) || defined(KOKKOS_ENABLE_SYCL)
private:
  friend KOKKOS_FUNCTION Box operator+(Box box, Box const &other)
  {
    return box += other;
  }
#endif
};

struct DiscretizedBox
{
  // Initialize all max coordinates with 0 and all min coordinates with max.
  KOKKOS_FUNCTION constexpr DiscretizedBox () :
    _data (single_max + (0lu<<10) + (single_max << 20) + (0lu << 30) + (single_max << 40) + (0lu << 50))
  {}

  KOKKOS_FUNCTION DiscretizedBox (Box const& local_box, Box const& global_box)
  {
    const Point& global_min_corner = global_box.minCorner();
    const Point& global_max_corner = global_box.maxCorner();
    const Point& local_min_corner = local_box.minCorner();
    const Point& local_max_corner = local_box.maxCorner();

    const double inv_h_x = (1 << 10) / (global_max_corner[0] - global_min_corner[0]);
    std::uint64_t min_x = std::max<std::uint64_t>(0, std::floor((local_min_corner[0]- global_min_corner[0]) * inv_h_x));
    std::uint64_t max_x = std::min<std::uint64_t>(single_max, std::ceil((local_max_corner[0] - global_min_corner[0]) * inv_h_x));

    const double inv_h_y = (1 << 10) / (global_max_corner[1] - global_min_corner[1]);
    std::uint64_t min_y = std::max<std::uint64_t>(0,std::floor((local_min_corner[1] - global_min_corner[1]) * inv_h_y));
    std::uint64_t max_y = std::min<std::uint64_t>(single_max, std::ceil((local_max_corner[1] - global_min_corner[1]) * inv_h_y));

    const double inv_h_z = (1 << 10) / (global_max_corner[2] - global_min_corner[2]);
    std::uint64_t min_z = std::max<std::uint64_t>(0,std::floor((local_min_corner[2] - global_min_corner[2]) * inv_h_z));
    std::uint64_t max_z = std::min<std::uint64_t>(single_max, std::ceil((local_max_corner[2] - global_min_corner[2]) * inv_h_z));

    _data = min_x + (max_x << 10) + (min_y << 20) + (max_y << 30) + (min_z << 40) + (max_z << 50);
  }

  KOKKOS_FUNCTION constexpr Box to_box(Box const& global_box) const
  {
    using KokkosExt::max;
    using KokkosExt::min;

    const Point& global_min_corner = global_box.minCorner();
    const Point& global_max_corner = global_box.maxCorner();

    const float h_x = (global_max_corner[0] - global_min_corner[0])/ (1<<10);
    const float h_y = (global_max_corner[1] - global_min_corner[1])/ (1<<10);
    const float h_z = (global_max_corner[2] - global_min_corner[2])/ (1<<10);

    float box_min_x = max(global_min_corner[0], global_min_corner[0]+(_data & min_x_mask)*h_x); 
    float box_max_x = min(global_max_corner[0], global_min_corner[0]+((_data & max_x_mask) >> 10)*h_x);
    float box_min_y = max(global_min_corner[1], global_min_corner[1]+((_data & min_y_mask) >> 20)*h_y);
    float box_max_y = min(global_max_corner[1], global_min_corner[1]+((_data & max_y_mask) >> 30)*h_y);
    float box_min_z = max(global_min_corner[2], global_min_corner[2]+((_data & min_z_mask) >> 40)*h_z);
    float box_max_z = min(global_max_corner[2], global_min_corner[2]+((_data & max_z_mask) >> 50)*h_z);
    return {{box_min_x, box_min_y, box_min_z}, {box_max_x, box_max_y, box_max_z}};
  }

  KOKKOS_FUNCTION constexpr DiscretizedBox &operator+=(DiscretizedBox const &other)
  {
    using KokkosExt::max;
    using KokkosExt::min;

    _data = min(_data & min_x_mask, other._data & min_x_mask) |
            max(_data & max_x_mask, other._data & max_x_mask) |
            min(_data & min_y_mask, other._data & min_y_mask) |
            max(_data & max_y_mask, other._data & max_y_mask) |
            min(_data & min_z_mask, other._data & min_z_mask) |
            max(_data & max_z_mask, other._data & max_z_mask);

    return *this;
  }

  private:
  std::uint64_t _data;
  static constexpr std::uint64_t single_max = (1lu << 10) -1;
  static constexpr std::uint64_t min_x_mask = single_max;
  static constexpr std::uint64_t max_x_mask = single_max << 10;
  static constexpr std::uint64_t min_y_mask = single_max << 20;
  static constexpr std::uint64_t max_y_mask = single_max << 30;
  static constexpr std::uint64_t min_z_mask = single_max << 40;
  static constexpr std::uint64_t max_z_mask = single_max << 50;
};

} // namespace ArborX

#endif
