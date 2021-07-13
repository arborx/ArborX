/****************************************************************************
 * Copyright (c) 2012-2019 by the ArborX authors                            *
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

#include <stdlib.h>
#include <unistd.h>

struct Dummy
{
  int count;
};

using MemorySpace = Kokkos::HostSpace;
using ExecutionSpace = Kokkos::DefaultHostExecutionSpace;

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

struct PrintfCallback
{
  Kokkos::View<int, ExecutionSpace, Kokkos::MemoryTraits<Kokkos::Atomic>> c_;
  PrintfCallback()
      : c_{"counter"}
  {
  }
  template <typename Predicate, typename OutputFunctor>
  KOKKOS_FUNCTION void operator()(Predicate const &predicate, int i,
                                  OutputFunctor const &out) const
  {
    int const j = getData(predicate);
    printf("%d callback (%d,%d)\n", ++c_(), i, j);
    out(i);
  }
};

template <typename T, typename... P>
std::vector<T> view2vec(Kokkos::View<T *, P...> view)
{
  std::vector<T> vec(view.size());
  Kokkos::deep_copy(Kokkos::View<T *, Kokkos::HostSpace,
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                        vec.data(), vec.size()),
                    view);
  return vec;
}

template <typename OutputView, typename OffsetView>
void print(OutputView const out, OffsetView const offset)
{
  int const n_queries = offset.extent(0) - 1;

  auto const h_out = view2vec(out);
  auto const h_offset = view2vec(offset);
  int count = 0;

  for (int j = 0; j < n_queries; ++j)
    for (int k = h_offset[j]; k < h_offset[j + 1]; ++k)
      printf("%d result (%d, %d)\n", ++count, h_out[k], j);
}

int main(int argc, char *argv[])
{
  Kokkos::ScopeGuard guard(argc, argv);

  int nqueries = 5, nprimitives = 5, nrepeats = 1;
  namespace bpo = boost::program_options;
  bpo::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ( "help", "help message" )
      ( "predicates", bpo::value<int>(&nqueries), "number of predicates" )
      ( "primitives", bpo::value<int>(&nprimitives), "number of primitives" )
      ( "iterations", bpo::value<int>(&nrepeats), "number of iterations" )
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

  ExecutionSpace space{};
  Dummy primitives{nprimitives};
  Dummy predicates{nqueries};

  for (int i = 0; i < nrepeats; i++)
  {
    int out_count;
    {
      Kokkos::Timer timer;
      ArborX::BoundingVolumeHierarchy<MemorySpace> bvh{space, primitives};

      Kokkos::View<int *, ExecutionSpace> indices("indices_ref", 0);
      Kokkos::View<int *, ExecutionSpace> offset("offset_ref", 0);
      bvh.query(
          space, predicates, ArborX::Details::DefaultCallback{}, indices,
          offset,
          ArborX::Experimental::TraversalPolicy{}.setPredicateSorting(true));

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
      brute.query(space, predicates, ArborX::Details::DefaultCallback{},
                  indices, offset);

      double time = timer.seconds();
      printf("Time BF: %lf\n", time);
      ARBORX_ASSERT(out_count == indices.extent(0));
    }
  }
  return 0;
}
