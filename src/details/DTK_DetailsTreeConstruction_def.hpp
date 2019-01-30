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

#ifndef DTK_DETAILS_TREE_CONSTRUCTION_DEF_HPP
#define DTK_DETAILS_TREE_CONSTRUCTION_DEF_HPP

#include "DTK_ConfigDefs.hpp"

#include <DTK_DBC.hpp>
#include <DTK_DetailsAlgorithms.hpp>
#include <DTK_DetailsUtils.hpp> // iota

#include <DTK_KokkosHelpers.hpp> // sgn, min, max

#include <Kokkos_Atomic.hpp>
#include <Kokkos_Sort.hpp>

#include <cassert>

namespace DataTransferKit
{
namespace Details
{

// nothing here

} // namespace Details
} // namespace DataTransferKit

// Explicit instantiation macro
#define DTK_TREECONSTRUCTION_INSTANT( NODE )                                   \
    namespace Details                                                          \
    {                                                                          \
    template struct TreeConstruction<typename NODE::device_type>;              \
    }

#endif
