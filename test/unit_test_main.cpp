/****************************************************************************
 * Copyright (c) 2012-2018 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <Kokkos_Core.hpp>

#include <Teuchos_GlobalMPISession.hpp>
#include <Teuchos_UnitTestRepository.hpp>

int main( int argc, char *argv[] )
{
    Teuchos::GlobalMPISession mpiSession( &argc, &argv );
    Teuchos::UnitTestRepository::setGloballyReduceTestResult( true );
    Kokkos::initialize( argc, argv );
    int return_val =
        Teuchos::UnitTestRepository::runUnitTestsFromMain( argc, argv );
    Kokkos::finalize();
    return return_val;
}
