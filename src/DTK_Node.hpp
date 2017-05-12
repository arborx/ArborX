/****************************************************************************
 * Copyright (c) 2012-2017 by the DataTransferKit authors                   *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the DataTransferKit library. DataTransferKit is     *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 ****************************************************************************/

#ifndef DTK_NODE_HPP
#define DTK_NODE_HPP

#include <DTK_Box.hpp>
#include <Kokkos_Pair.hpp>

namespace DataTransferKit
{
struct Node
{
    KOKKOS_INLINE_FUNCTION
    Node()
        : parent( nullptr )
        , children( {nullptr, nullptr} )
    {
    }

    Node *parent = nullptr;
    Kokkos::pair<Node *, Node *> children;
    BBox bounding_box;
};
}

#endif
