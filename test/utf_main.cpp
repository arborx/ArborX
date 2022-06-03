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

#include <Kokkos_Core.hpp>

#define BOOST_TEST_NO_MAIN
#include <boost/test/unit_test.hpp>

#if defined(ARBORX_MPI_UNIT_TEST)
#include <mpi.h>
#endif

struct ExecutionEnvironmentScopeGuard
{
  ExecutionEnvironmentScopeGuard(int argc, char *argv[])
  {
#if defined(ARBORX_MPI_UNIT_TEST)
    MPI_Init(&argc, &argv);
#endif
    Kokkos::initialize(argc, argv);
  }
  ~ExecutionEnvironmentScopeGuard()
  {
    Kokkos::finalize();
#if defined(ARBORX_MPI_UNIT_TEST)
    MPI_Finalize();
#endif
  }
};

bool init_function() { return true; }

int main(int argc, char *argv[])
{
  ExecutionEnvironmentScopeGuard scope_guard(argc, argv);
  return boost::unit_test::unit_test_main(&init_function, argc, argv);
}
