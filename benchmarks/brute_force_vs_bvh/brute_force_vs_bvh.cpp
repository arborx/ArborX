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
#include <ArborX_HyperBox.hpp>
#include <ArborX_HyperPoint.hpp>
#include <ArborX_HyperSphere.hpp>
#include <ArborX_LinearBVH.hpp>

#include <Kokkos_Core.hpp>

#include <boost/program_options.hpp>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;

template <int DIM>
struct Dummy
{
  int count;
};

// Primitives are a set of points located at (i, i, i),
// with i = 0, ..., n-1
template <int DIM>
struct ArborX::AccessTraits<Dummy<DIM>, ArborX::PrimitivesTag>
{
  using memory_space = MemorySpace;
  using size_type = typename MemorySpace::size_type;
  static KOKKOS_FUNCTION size_type size(Dummy<DIM> d) { return d.count; }
  static KOKKOS_FUNCTION auto get(Dummy<DIM>, size_type i)
  {
    ArborX::ExperimentalHyperGeometry::Point<DIM> point;
    for (int d = 0; d < DIM; ++d)
      point[d] = i;
    return point;
  }
};

// Predicates are sphere intersections with spheres of radius i
// centered at (i, i, i), with i = 0, ..., n-1
template <int DIM>
struct ArborX::AccessTraits<Dummy<DIM>, ArborX::PredicatesTag>
{
  using memory_space = MemorySpace;
  using size_type = typename MemorySpace::size_type;
  static KOKKOS_FUNCTION size_type size(Dummy<DIM> d) { return d.count; }
  static KOKKOS_FUNCTION auto get(Dummy<DIM>, size_type i)
  {
    ArborX::ExperimentalHyperGeometry::Point<DIM> center;
    for (int d = 0; d < DIM; ++d)
      center[d] = i;
    return attach(intersects(ArborX::ExperimentalHyperGeometry::Sphere<DIM>{
                      center, (float)i}),
                  i);
  }
};

template <int DIM>
void run(int nprimitives, int nqueries, int nrepeats)
{
  ExecutionSpace space{};
  Dummy<DIM> primitives{nprimitives};
  Dummy<DIM> predicates{nqueries};

  printf("Dimension : %d\n", DIM);
  printf("Primitives: %d\n", nprimitives);
  printf("Predicates: %d\n", nqueries);
  printf("Iterations: %d\n", nrepeats);

  using Box = ArborX::ExperimentalHyperGeometry::Box<DIM>;

  for (int i = 0; i < nrepeats; i++)
  {
    unsigned int out_count;
    {
      Kokkos::Timer timer;
      ArborX::BasicBoundingVolumeHierarchy<MemorySpace, Box> bvh{space,
                                                                 primitives};

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
      ArborX::BruteForce<MemorySpace, Box> brute{space, primitives};

      Kokkos::View<int *, ExecutionSpace> indices("indices", 0);
      Kokkos::View<int *, ExecutionSpace> offset("offset", 0);
      brute.query(space, predicates, indices, offset);

      space.fence();
      double time = timer.seconds();
      printf("Time BF: %lf\n", time);
      ARBORX_ASSERT(out_count == indices.extent(0));
    }
  }
}

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  int dim;
  int nprimitives;
  int nqueries;
  int nrepeats;
  namespace bpo = boost::program_options;
  bpo::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ( "help", "help message" )
      ( "dimension", bpo::value<int>(&dim)->default_value(3), "dimension" )
      ( "predicates", bpo::value<int>(&nqueries)->default_value(5), "number of predicates" )
      ( "primitives", bpo::value<int>(&nprimitives)->default_value(5), "number of primitives" )
      ( "repetitions", bpo::value<int>(&nrepeats)->default_value(1), "number of repetitions" )
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
  ARBORX_ASSERT(nprimitives > 0);
  ARBORX_ASSERT(nqueries > 0);

  switch (dim)
  {
  case 3:
    run<3>(nprimitives, nqueries, nrepeats);
    break;
#if KOKKOS_VERSION >= 30700
  case 2:
    run<2>(nprimitives, nqueries, nrepeats);
    break;
  case 4:
    run<4>(nprimitives, nqueries, nrepeats);
    break;
  case 5:
    run<5>(nprimitives, nqueries, nrepeats);
    break;
  case 6:
    run<6>(nprimitives, nqueries, nrepeats);
    break;
#endif
  default:
    std::cerr << "Dimension " << dim << " not supported.\n";
    return 1;
  }

  return 0;
}
