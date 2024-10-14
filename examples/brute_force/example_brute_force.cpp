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

#include <ArborX.hpp>

#include <Kokkos_Core.hpp>

#include <iostream>

struct Dummy
{
  int count;
};

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;

template <typename MemorySpace, typename Index = int>
struct Iota
{
  using memory_space = MemorySpace;
  using index_type = Index;

  size_t _n;

  template <typename T,
            typename Enable = std::enable_if_t<std::is_integral_v<T>>>
  Iota(T n)
      : _n(n)
  {}
};

template <typename MemorySpace>
struct ArborX::AccessTraits<Iota<MemorySpace>, ArborX::PrimitivesTag>
{
  using Self = Iota<MemorySpace>;

  using memory_space = typename Self::memory_space;
  static KOKKOS_FUNCTION size_t size(Self const &self) { return self._n; }
  static KOKKOS_FUNCTION auto get(Self const &, size_t i)
  {
    return (typename Self::index_type)i;
  }
};

struct DummyIndexableGetter
{
  int count;

  using memory_space = MemorySpace;
  KOKKOS_FUNCTION auto size() const { return count; }
  KOKKOS_FUNCTION auto operator()(int i) const
  {
    return ArborX::Point{(float)i, (float)i, (float)i};
  }
};

template <>
struct ArborX::AccessTraits<Dummy, ArborX::PredicatesTag>
{
  using memory_space = MemorySpace;
  using size_type = typename MemorySpace::size_type;

  static KOKKOS_FUNCTION size_type size(Dummy const &d) { return d.count; }
  static KOKKOS_FUNCTION auto get(Dummy const &, size_type i)
  {
    ArborX::Point center{(float)i, (float)i, (float)i};
    return ArborX::intersects(Sphere{center, (float)i});
  }
};

template <typename View,
          typename Enable = std::enable_if_t<Kokkos::is_view_v<View>>>
std::ostream &operator<<(std::ostream &os, View const &view)
{
  auto view_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view);
  std::copy(view_host.data(), view_host.data() + view.size(),
            std::ostream_iterator<typename View::value_type>(std::cout, " "));
  return os;
}

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  ExecutionSpace space{};

  int nprimitives = 5;
  int npredicates = 5;

  Iota<MemorySpace> primitives{nprimitives};
  DummyIndexableGetter indexable_getter{nprimitives};
  Dummy predicates{npredicates};

  unsigned int out_count;
  {
    ArborX::BoundingVolumeHierarchy bvh{space, primitives, indexable_getter};

    Kokkos::View<int *, ExecutionSpace> indices("Example::indices_ref", 0);
    Kokkos::View<int *, ExecutionSpace> offset("Example::offset_ref", 0);
    bvh.query(space, predicates, indices, offset);

    out_count = indices.extent(0);

    std::cout << "offset (bvh): " << offset << std::endl;
    std::cout << "indices (bvh): " << indices << std::endl;
  }

  {
    ArborX::BruteForce brute{space, primitives, indexable_getter};

    Kokkos::View<int *, ExecutionSpace> indices("Example::indices", 0);
    Kokkos::View<int *, ExecutionSpace> offset("Example::offset", 0);
    brute.query(space, predicates, indices, offset);

    // The offset output should match the one from bvh. The indices output
    // should have the same indices for each offset entry, but they may be
    // in a different order.
    std::cout << "offset (bf): " << offset << std::endl;
    std::cout << "indices (bf): " << indices << std::endl;

    if (indices.extent(0) != out_count)
      Kokkos::abort("The sizes of indices do not match");
  }

  return 0;
}
