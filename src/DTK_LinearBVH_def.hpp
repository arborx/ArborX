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

#ifndef DTK_LINEAR_BVH_DEF_HPP
#define DTK_LINEAR_BVH_DEF_HPP

namespace DataTransferKit
{

// FIXME nothing here...

} // namespace DataTransferKit

// Explicit instantiation macro
#define DTK_LINEAR_BVH_INSTANT( NODE )                                         \
    template class BoundingVolumeHierarchy<typename NODE::device_type>;

#endif
