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

#ifndef ARBORX_IO_HPP
#define ARBORX_IO_HPP

#include <ArborX_AccessTraits.hpp>
#include <ArborX_Box.hpp>
#include <ArborX_Exception.hpp>
#include <ArborX_Point.hpp>

#include <Kokkos_Core.hpp>

#include <fstream>
#include <iomanip>
#include <limits>

namespace ArborX
{

template <typename T>
struct IOTraits
{
  using not_specialized = void; // tag to detect existence of a specialization
};

template <typename Traits>
using IOTraitsNotSpecializedArchetypeAlias = typename Traits::not_specialized;

template <>
struct IOTraits<Point>
{
  static std::ostream &write(std::ostream &os, Point const &point, bool binary)
  {
    if (!binary)
    {
      std::ios_base::fmtflags flags(os.flags());

      using T = std::decay_t<decltype(std::declval<Point>()[0])>;
      os << std::fixed
         << std::setprecision(std::numeric_limits<T>::max_digits10);
      os << point[0] << " " << point[1] << " " << point[2];

      os.flags(flags);
    }
    else
      os.write(reinterpret_cast<char const *>(&point), sizeof(Point));

    return os;
  }

  static std::istream &read(std::istream &is, Point &point, bool binary)
  {
    if (!binary)
      is >> point[0] >> point[1] >> point[2];
    else
      is.read(reinterpret_cast<char *>(&point), sizeof(Point));

    return is;
  }
};

template <>
struct IOTraits<Box>
{
  static std::ostream &write(std::ostream &os, Box const &box, bool binary)
  {
    if (!binary)
    {
      IOTraits<Point>::write(os, box.minCorner(), binary);
      os << " ";
      IOTraits<Point>::write(os, box.maxCorner(), binary);
    }
    else
    {
      os.write(reinterpret_cast<char const *>(&box), sizeof(Box));
    }
    return os;
  }

  static std::istream &read(std::istream &is, Box &box, bool binary)
  {
    if (!binary)
    {
      IOTraits<Point>::read(is, box.minCorner(), binary);
      IOTraits<Point>::read(is, box.maxCorner(), binary);
    }
    else
    {
      is.read(reinterpret_cast<char *>(&box), sizeof(Box));
    }

    return is;
  }
};

template <typename Primitives>
void saveData(std::string const &filename, Primitives const &primitives,
              bool binary = true)
{
  Details::check_valid_access_traits(PrimitivesTag{}, primitives);

  using Access = AccessTraits<Primitives, PrimitivesTag>;
  using MemorySpace = typename Access::memory_space;
  using ExecutionSpace = typename MemorySpace::execution_space;

  std::ofstream output;
  if (!binary)
    output.open(filename);
  else
    output.open(filename, std::ofstream::binary);
  ARBORX_ASSERT(output.good());

  int n = Access::size(primitives);

  using Geometry = std::decay_t<decltype(Access::get(std::declval<Primitives>(),
                                                     std::declval<size_t>()))>;

  // Construct a primitives view even if the original primitives was already a
  // Kokkos::View. This is the safest thing to do, as it cannot be guaranteed
  // that the AccessTraits were not specialized for that view.
  Kokkos::View<Geometry *, MemorySpace> primitives_view(
      Kokkos::ViewAllocateWithoutInitializing("ArborX::primitives"), n);
  Kokkos::parallel_for(
      "ArborX::IO::build_primitives_view",
      Kokkos::RangePolicy<ExecutionSpace>(ExecutionSpace{}, 0, n),
      KOKKOS_LAMBDA(int const i) {
        primitives_view(i) = Access::get(primitives, i);
      });
  auto primitives_view_host = Kokkos::create_mirror_view(primitives_view);

  int const dim = 3;
  if (!binary)
    output << n << " " << dim << "\n";
  else
  {
    output.write(reinterpret_cast<char const *>(&n), sizeof(int));
    output.write(reinterpret_cast<char const *>(&dim), sizeof(int));
  }

  for (int i = 0; i < n; ++i)
  {
    IOTraits<Geometry>::write(output, primitives_view_host(i), binary);
    if (!binary)
      output << "\n";
  }

  output.close();
}

template <typename MemorySpace, typename Geometry>
auto loadData(std::string const &filename, bool binary)
{
  std::ifstream input;
  if (!binary)
    input.open(filename);
  else
    input.open(filename, std::ifstream::binary);
  ARBORX_ASSERT(input.good());

  int n = 0;
  int dim;
  if (!binary)
  {
    input >> n >> dim;
  }
  else
  {
    input.read(reinterpret_cast<char *>(&n), sizeof(int));
    input.read(reinterpret_cast<char *>(&dim), sizeof(int));
  }
  ARBORX_ASSERT(n > 0);
  ARBORX_ASSERT(dim == 3);

  Kokkos::View<Geometry *, MemorySpace> data(
      Kokkos::ViewAllocateWithoutInitializing("ArborX::data"), n);
  auto data_host = Kokkos::create_mirror_view(data);

  for (int i = 0; i < n; ++i)
    IOTraits<Geometry>::read(input, data(i), binary);
  input.close();

  Kokkos::deep_copy(data, data_host);

  return data;
}

} // namespace ArborX

#endif
