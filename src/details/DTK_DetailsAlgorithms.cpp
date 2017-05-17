/****************************************************************************
 * Copyright (c) 2012-2017 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 ****************************************************************************/
#include <DTK_DetailsAlgorithms.hpp>

#include <cmath>

namespace DataTransferKit
{
namespace Details
{

void expand( Box &box, Point const &point )
{
    for ( int d = 0; d < 3; ++d )
    {
        if ( point[d] < box[2 * d + 0] )
            box[2 * d + 0] = point[d];
        if ( point[d] > box[2 * d + 1] )
            box[2 * d + 1] = point[d];
    }
}

} // end namespace Details
} // end namespace DataTransferKit
