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

#ifndef DTK_NODE_HPP
#define DTK_NODE_HPP

#include <DTK_Box.hpp>
#include <Kokkos_Pair.hpp>

namespace DataTransferKit
{
struct Node
{
    KOKKOS_INLINE_FUNCTION
    Node() = default;

    KOKKOS_INLINE_FUNCTION
    Node( const Kokkos::pair<Node *, Node *> &c, const Box &bb )
        : children( c )
        , bounding_box( bb )
    {
    }

    Kokkos::pair<Node *, Node *> children = {nullptr, nullptr};
    Box bounding_box;
};
} // namespace DataTransferKit

#endif
