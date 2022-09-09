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

#include "brute_force_vs_bvh.hpp"

#include <Kokkos_Core.hpp>

#include <boost/program_options.hpp>

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
  assert(nprimitives > 0);
  assert(nqueries > 0);

  using ArborXBenchmark::run;

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
