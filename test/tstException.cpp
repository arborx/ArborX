/****************************************************************************
 * Copyright (c) 2012-2019 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <DTK_Search_Exception.hpp>

#include "Teuchos_UnitTestHarness.hpp"

// Check that a DataTransferKit::SearchException looks different than a
// std::runtime_error as it inherits from std::logic_error.
TEUCHOS_UNIT_TEST( SearchException, differentiation_test )
{
    try
    {
        throw std::runtime_error( "runtime error" );
    }
    catch ( const DataTransferKit::SearchException &assertion )
    {
        TEST_ASSERT( 0 );
    }
    catch ( ... )
    {
        TEST_ASSERT( 1 );
    }
}

// Check that a DataTransferKit::SearchException can be caught and the
// appropriate error message is written.
TEUCHOS_UNIT_TEST( SearchException, message_test )
{
    std::string message;

    try
    {
        throw DataTransferKit::SearchException( "cond" );
    }
    catch ( const DataTransferKit::SearchException &assertion )
    {
        message = std::string( assertion.what() );
    }
    catch ( ... )
    {
        TEST_ASSERT( 0 );
    }

    const std::string true_message( "DTK Search exception: cond" );
    TEST_ASSERT( 0 == message.compare( true_message ) );
}

// Test the assertion check
TEUCHOS_UNIT_TEST( SearchException, assertion_test )
{
    try
    {
        DTK_SEARCH_ASSERT( 0 );
        throw std::runtime_error( "this shouldn't be thrown" );
    }
    catch ( const DataTransferKit::SearchException &assertion )
    {
        std::string message( assertion.what() );
        std::string true_message( "DTK Search exception: 0, failed at" );
        std::string::size_type idx = message.find( true_message );
        if ( idx == std::string::npos )
        {
            TEST_ASSERT( 0 );
        }
    }
    catch ( ... )
    {
        TEST_ASSERT( 0 );
    }
}
