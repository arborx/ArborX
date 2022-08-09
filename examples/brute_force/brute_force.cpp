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

#include <boost/program_options.hpp>

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
    return attach(
        intersects(Sphere{{{(float)i, (float)i, (float)i}}, (float)i}), i);
  }
};

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  int nqueries;
  int nprimitives;
  int nrepeats;
  namespace bpo = boost::program_options;
  bpo::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ( "help", "help message" )
      ( "predicates", bpo::value<int>(&nqueries)->default_value(5), "number of predicates" )
      ( "primitives", bpo::value<int>(&nprimitives)->default_value(5), "number of primitives" )
      ( "iterations", bpo::value<int>(&nrepeats)->default_value(1), "number of iterations" )
      ;
  // clang-format on
  bpo::variables_map vm;
  bpo::store(bpo::command_line_parser(argc, argv).options(desc).run(), vm);
  bpo::notify(vm);

  if (vm.count("help") > 0)
  {
    std::cout << desc << '\n';
    return 1;
  }
  printf("Primitives: %d\n", nprimitives);
  printf("Predicates: %d\n", nqueries);
  printf("Iterations: %d\n", nrepeats);

  ARBORX_ASSERT(nprimitives > 0);
  ARBORX_ASSERT(nqueries > 0);

  ExecutionSpace space{};
  Dummy primitives{nprimitives};
  Dummy predicates{nqueries};

  for (int i = 0; i < nrepeats; i++)
  {
    unsigned int out_count;
    {
      Kokkos::Timer timer;
      ArborX::BoundingVolumeHierarchy<MemorySpace> bvh{space, primitives};

      Kokkos::View<int *, ExecutionSpace> indices("indices_ref", 0);
      Kokkos::View<int *, ExecutionSpace> offset("offset_ref", 0);
      bvh.query(space, predicates, indices, offset);

      space.fence();
      double time = timer.seconds();
      if (i == 0)
        printf("Collisions: %.5f\n",
               (float)(indices.extent(0)) / (nprimitives * nqueries));
      printf("Time BVH: %lf\n", time);
      out_count = indices.extent(0);
    }

    {
      Kokkos::Timer timer;
      ArborX::BruteForce<MemorySpace> brute{space, primitives};

      Kokkos::View<int *, ExecutionSpace> indices("indices", 0);
      Kokkos::View<int *, ExecutionSpace> offset("offset", 0);
      brute.query(space, predicates, indices, offset);

      space.fence();
      double time = timer.seconds();
      printf("Time BF: %lf\n", time);
      ARBORX_ASSERT(out_count == indices.extent(0));
    }
  }
  return 0;
}
