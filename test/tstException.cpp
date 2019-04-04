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

#include <DTK_Search_Exception.hpp>

#include <boost/test/unit_test.hpp>

#define BOOST_TEST_MODULE DesignByContract

BOOST_AUTO_TEST_CASE( dumb )
{
    using namespace DataTransferKit;
    BOOST_CHECK_NO_THROW( DTK_SEARCH_ASSERT( true ) );
    BOOST_CHECK_THROW( DTK_SEARCH_ASSERT( false ), SearchException );
    std::string const prefix = "DTK Search exception: ";
    std::string const message = "Keep calm and chive on!";
    BOOST_CHECK_EXCEPTION( throw SearchException( message ), SearchException,
                           [&]( SearchException const &e ) {
                               return prefix + message == e.what();
                           } );
}
