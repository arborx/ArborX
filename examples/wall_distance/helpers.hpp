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

#ifndef ARBORX_EXAMPLE_HELPERS_HPP
#define ARBORX_EXAMPLE_HELPERS_HPP

#include <string>
#include <vector>

#include <Panzer_STK_Interface.hpp>
#include <Panzer_Workset.hpp>
#include <Teuchos_RCP.hpp>
#include <mpi.h>

enum class DistanceType
{
  NODE,
  CELL
};

constexpr int workset_size = 128;

bool check_names(MPI_Comm comm, std::string const &filename,
                 std::vector<std::string> const &block_names,
                 std::vector<std::string> const &wall_names);

Teuchos::RCP<panzer_stk::STK_Interface>
build_mesh(MPI_Comm comm, std::string const &filename,
           std::vector<std::string> &block_names,
           std::string const &distance_field_name, DistanceType distance_type,
           int restart_index);

std::vector<panzer::Workset>
build_worksets(Teuchos::RCP<panzer_stk::STK_Interface> const &mesh,
               std::vector<std::string> const &block_names,
               std::string const &basis_type, int const basis_order,
               int int_order);
#endif
