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
#ifndef ARBORX_SEARCH_EXCEPTION_HPP
#define ARBORX_SEARCH_EXCEPTION_HPP

#include <stdexcept>
#include <string>

namespace DataTransferKit
{
class SearchException : public std::logic_error
{
  public:
    SearchException( std::string const &msg )
        : std::logic_error( std::string( "DTK Search exception: " ) + msg )
    {
    }

    virtual ~SearchException() throw() {}
};

} // namespace DataTransferKit

#define ARBORX_STRINGIZE_DETAIL( x ) #x
#define ARBORX_STRINGIZE( x ) ARBORX_STRINGIZE_DETAIL( x )

// FIXME: Unconditionally assert for now
// Once moved out, possibly make it conditional
#define ARBORX_ASSERT( c )                                                     \
    if ( !( c ) )                                                              \
    throw DataTransferKit::SearchException(                                    \
        #c ", failed at " __FILE__ ":" ARBORX_STRINGIZE( __LINE__ ) "." )

#endif
