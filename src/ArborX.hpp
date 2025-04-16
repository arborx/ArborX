/****************************************************************************
 * Copyright (c) 2025, ArborX authors                                       *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef ARBORX_HPP
#define ARBORX_HPP

#include <ArborX_Config.hpp>

// Indexes
#include <ArborX_BruteForce.hpp>
#ifdef ARBORX_ENABLE_MPI
#include <ArborX_DistributedTree.hpp>
#endif
#include <ArborX_CrsGraphWrapper.hpp>
#include <ArborX_LinearBVH.hpp>

// Indexes helpers
#include <detail/ArborX_AttachIndices.hpp>
#include <detail/ArborX_NeighborList.hpp>
#include <detail/ArborX_PredicateHelpers.hpp>
#include <detail/ArborX_Predicates.hpp>

// Misc
#include <misc/ArborX_Exception.hpp>

#endif
