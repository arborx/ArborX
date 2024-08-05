/****************************************************************************
 * Copyright (c) 2017-2022 by the ArborX authors                            *
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

#include <ArborX_Box.hpp>
#include <ArborX_BruteForce.hpp>
#ifdef ARBORX_ENABLE_MPI
#include <ArborX_DistributedTree.hpp>
#endif
#include <ArborX_CrsGraphWrapper.hpp>
#include <ArborX_Exception.hpp>
#include <ArborX_LinearBVH.hpp>
#include <ArborX_OBB.hpp>
#include <ArborX_Point.hpp>
#include <ArborX_PredicateHelpers.hpp>
#include <ArborX_Predicates.hpp>
#include <ArborX_Sphere.hpp>
// FIXME: we include ArborX_DetailsUtils.hpp only for backward compatibility for
// users using deprecated functions in ArborX namespace (min, max,
// adjacentDifference, ...). This header should be removed when we remove those
// functions.
#include <ArborX_DetailsUtils.hpp>

#endif
