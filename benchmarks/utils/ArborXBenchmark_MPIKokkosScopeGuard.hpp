/****************************************************************************
 * Copyright (c) 2025, ArborX authors                                       *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_BENCHMARK_MPI_KOKKOS_SCOPE_GUARD_HPP
#define ARBORX_BENCHMARK_MPI_KOKKOS_SCOPE_GUARD_HPP

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstdlib>
#include <string>

#include <mpi.h>

namespace ArborXBenchmark
{

class MPIKokkosScopeGuard
{
public:
  MPIKokkosScopeGuard(int &argc, char *argv[])
  {
    MPI_Init(&argc, &argv);

    int comm_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

    bool const is_kokkos_help_present =
        std::any_of(argv, argv + argc,
                    [](std::string const &x) { return x == "--kokkos-help"; });
    if (is_kokkos_help_present)
    {
      if (comm_rank == 0)
      {
        std::atexit([]() {
          int is_finalized;
          MPI_Finalized(&is_finalized);
          if (is_finalized == 0)
            MPI_Finalize();
        });
        Kokkos::initialize(argc, argv);
        Kokkos::finalize();
      }
      MPI_Finalize();
      std::exit(EXIT_SUCCESS);
    }

    // We assume at most one instance of the "--help" argument in the command
    // line
    auto *help_it = std::find_if(
        argv, argv + argc, [](std::string const &x) { return x == "--help"; });
    auto is_help_present = (help_it != argv + argc);
    char *help_ptr = is_help_present ? *help_it : nullptr;
    if (is_help_present && comm_rank != 0)
    {
      // Shift the remainder of the argv list by one. Note that argv has
      // (argc + 1) arguments, the last one always being nullptr. The following
      // loop moves the trailing nullptr element as well
      for (int k = help_it - argv; k < argc; ++k)
        argv[k] = argv[k + 1];
      --argc;
    }
    Kokkos::initialize(argc, argv);
    if (is_help_present && comm_rank != 0)
    {
      ++argc;
      argv[argc - 1] = help_ptr;
      argv[argc] = nullptr;
    }
  }

  MPIKokkosScopeGuard(MPIKokkosScopeGuard const &) = delete;
  MPIKokkosScopeGuard &operator=(MPIKokkosScopeGuard const &) = delete;
  MPIKokkosScopeGuard(MPIKokkosScopeGuard &&) = delete;
  MPIKokkosScopeGuard &operator=(MPIKokkosScopeGuard &&) = delete;

  ~MPIKokkosScopeGuard()
  {
    Kokkos::finalize();
    MPI_Finalize();
  }
};

} // namespace ArborXBenchmark

#endif
