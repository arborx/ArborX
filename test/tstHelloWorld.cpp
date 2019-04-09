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

#include <Kokkos_Core.hpp>

#include <boost/test/unit_test.hpp>

#include <mpi.h>

#define BOOST_TEST_MODULE HelloWorld

BOOST_AUTO_TEST_CASE( mpi )
{
    // Get the number of processes
    int world_size;
    MPI_Comm_size( MPI_COMM_WORLD, &world_size );

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &world_rank );

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name( processor_name, &name_len );

    // Print off a hello world message
    printf( "Hello world from processor %s, rank %d out of %d processors\n",
            processor_name, world_rank, world_size );
}

BOOST_AUTO_TEST_CASE( kokkos )
{
    Kokkos::parallel_for( 15, KOKKOS_LAMBDA( const int i ) {
        // printf works in a CUDA parallel kernel; std::ostream does not.
        printf( "Hello from i = %i\n", i );
    } );
}
