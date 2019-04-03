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
#ifndef DTK_SEARCH_EXCEPTION_HPP
#define DTK_SEARCH_EXCEPTION_HPP

#include <stdexcept>
#include <string>

// FIXME
// This dependency on DTK will have to be resolved later.
// Particularly, HAVE_DTK_DBC macros and DTK_MARK_REGION
#include <DTK_ConfigDefs.hpp>

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

#define DTK_SEARCH_STRINGIZE_DETAIL( x ) #x
#define DTK_SEARCH_STRINGIZE( x ) DTK_SEARCH_STRINGIZE_DETAIL( x )

#if HAVE_DTK_DBC
#define DTK_SEARCH_ASSERT( c )                                                 \
    if ( !( c ) )                                                              \
    throw DataTransferKit::SearchException(                                    \
        #c ", failed at " __FILE__ ":" DTK_SEARCH_STRINGIZE( __LINE__ ) "." )
#else
#define DTK_SEARCH_ASSERT( c )
#endif

#endif
