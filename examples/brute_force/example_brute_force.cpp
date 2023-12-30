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

#include <ArborX_BruteForce.hpp>
#include <ArborX_LinearBVH.hpp>
#include <ArborX_Sphere.hpp>

#include <Kokkos_Core.hpp>

struct Dummy
{
  int count;
};

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;

template <>
struct ArborX::AccessTraits<Dummy, ArborX::PrimitivesTag>
{
  using memory_space = MemorySpace;
  using size_type = typename MemorySpace::size_type;
  static KOKKOS_FUNCTION size_type size(Dummy const &d) { return d.count; }
  static KOKKOS_FUNCTION Point get(Dummy const &, size_type i)
  {
    return {{(float)i, (float)i, (float)i}};
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
    float radius = i;
    return ArborX::intersects(Sphere{center, radius});
  }
};

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  ExecutionSpace space{};

  int nprimitives = 5;
  int npredicates = 5;

  Dummy primitives{nprimitives};
  Dummy predicates{npredicates};

  unsigned int out_count;
  {
    ArborX::BoundingVolumeHierarchy<MemorySpace> bvh{space, primitives};

    Kokkos::View<int *, ExecutionSpace> indices("Example::indices_ref", 0);
    Kokkos::View<int *, ExecutionSpace> offset("Example::offset_ref", 0);
    bvh.query(space, predicates, indices, offset);

    out_count = indices.extent(0);
  }

  {
    ArborX::BruteForce<MemorySpace> brute{space, primitives};

    Kokkos::View<int *, ExecutionSpace> indices("Example::indices", 0);
    Kokkos::View<int *, ExecutionSpace> offset("Example::offset", 0);
    brute.query(space, predicates, indices, offset);

    ARBORX_ASSERT(out_count == indices.extent(0));
  }

  return 0;
}
