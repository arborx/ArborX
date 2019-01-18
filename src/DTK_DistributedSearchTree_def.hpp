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

#ifndef DTK_DISTRIBUTED_SEARCH_TREE_DEF_HPP
#define DTK_DISTRIBUTED_SEARCH_TREE_DEF_HPP

namespace DataTransferKit
{

// FIXME nothing here...

} // namespace DataTransferKit

// Explicit instantiation macro
#define DTK_DISTRIBUTED_SEARCH_TREE_INSTANT( NODE )                            \
    template class DistributedSearchTree<typename NODE::device_type>;

#endif
