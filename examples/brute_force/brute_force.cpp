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

#include <Kokkos_Core.hpp>

#include <boost/program_options.hpp>

template <int DIM>
struct Dummy
{
  int count;
};

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecutionSpace::memory_space;

template <int DIM>
struct ArborX::AccessTraits<Dummy<DIM>, ArborX::PrimitivesTag>
{
  using memory_space = MemorySpace;
  using size_type = typename MemorySpace::size_type;
  static KOKKOS_FUNCTION size_type size(Dummy<DIM> const &d) { return d.count; }
  static KOKKOS_FUNCTION PointD<DIM> get(Dummy<DIM> const &, size_type i)
  {
    PointD<DIM> point;
    for (int d = 0; d < DIM; ++d)
      point[d] = (float)i;
    return point;
  }
};

template <int DIM>
struct ArborX::AccessTraits<Dummy<DIM>, ArborX::PredicatesTag>
{
  using memory_space = MemorySpace;
  using size_type = typename MemorySpace::size_type;
  static KOKKOS_FUNCTION size_type size(Dummy<DIM> const &d) { return d.count; }
  static KOKKOS_FUNCTION auto get(Dummy<DIM> const &, size_type i)
  {
    PointD<DIM> center;
    for (int d = 0; d < DIM; ++d)
      center[d] = (float)i;
    return attach(intersects(SphereD<DIM>{center, (float)i}), i);
  }
};

template <int DIM>
void run(int nprimitives, int nqueries)
{
  ExecutionSpace space{};
  Dummy<DIM> primitives{nprimitives};
  Dummy<DIM> predicates{nqueries};

#if 0
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
#endif

  {
    Kokkos::Timer timer;
    ArborX::BruteForce<MemorySpace, DIM> brute{space, primitives};

    Kokkos::View<int *, ExecutionSpace> indices("indices", 0);
    Kokkos::View<int *, ExecutionSpace> offset("offset", 0);
    brute.query(space, predicates, indices, offset);

    space.fence();
    double time = timer.seconds();
    printf("Time BF: %lf\n", time);
    // ARBORX_ASSERT(out_count == indices.extent(0));
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
      ( "iterations", bpo::value<int>(&nrepeats)->default_value(1), "number of iterations" )
      ( "predicates", bpo::value<int>(&nqueries)->default_value(5), "number of predicates" )
      ( "primitives", bpo::value<int>(&nprimitives)->default_value(5), "number of primitives" )
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
  printf("Dimension : %d\n", dim);
  printf("Primitives: %d\n", nprimitives);
  printf("Predicates: %d\n", nqueries);
  printf("Iterations: %d\n", nrepeats);

  if (dim < 2 || dim > 3)
  {
    std::cerr << "Only dimensions 2-3 are allowed" << std::endl;
    return 1;
  }

  ARBORX_ASSERT(nprimitives > 0);
  ARBORX_ASSERT(nqueries > 0);

  for (int i = 0; i < nrepeats; i++)
  {
    switch (dim)
    {
    case 2:
      run<2>(nprimitives, nqueries);
      break;
    case 3:
      run<3>(nprimitives, nqueries);
      break;
    }
  }
  return 0;
}
