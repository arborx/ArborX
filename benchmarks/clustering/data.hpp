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
#ifndef DATA_HPP
#define DATA_HPP

#include <ArborX_Point.hpp>

#include <Kokkos_Core.hpp>

#include "parameters.hpp"

int getDataDimension(std::string const &filename, bool binary);

template <int DIM, typename MemorySpace>
Kokkos::View<ArborX::Point<DIM> *, MemorySpace>
loadData(ArborXBenchmark::Parameters const &params);

#endif
