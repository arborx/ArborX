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

#include "helpers.hpp"

#include <Ionit_Initializer.h>
#include <Ioss_DatabaseIO.h>
#include <Ioss_ElementBlock.h>
#include <Ioss_IOFactory.h>
#include <Ioss_Region.h>
#include <Ioss_SideSet.h>
#include <Panzer_STK_ExodusReaderFactory.hpp>
#include <Panzer_STK_WorksetFactory.hpp>
#include <Panzer_WorksetContainer.hpp>
#include <Panzer_WorksetNeeds.hpp>
#include <Teuchos_RCPStdSharedPtrConversions.hpp>
#include <mpi.h>

class STKMeshFactory : public panzer_stk::STK_ExodusReaderFactory
{
  DistanceType _distance_type;

public:
  STKMeshFactory(std::string const &file_name, DistanceType distance_type,
                 int const restart_index)
      : panzer_stk::STK_ExodusReaderFactory(file_name, restart_index)
      , _distance_type(distance_type)
  {}

  void
  completeMeshConstruction(panzer_stk::STK_Interface &mesh,
                           stk::ParallelMachine parallelMach) const override
  {
    if (!mesh.isInitialized())
      mesh.initialize(parallelMach, true, false);

    stk::mesh::MetaData &metaData = *mesh.getMetaData();
    stk::io::StkMeshIoBroker *meshData = const_cast<stk::io::StkMeshIoBroker *>(
        metaData.get_attribute<stk::io::StkMeshIoBroker>());
    TEUCHOS_ASSERT(metaData.remove_attribute(meshData));

    meshData->populate_bulk_data();

    int restartIndex = restartIndex_;
    if (restartIndex < 0)
      restartIndex = 1 + restartIndex +
                     meshData->get_input_ioss_region()->get_max_time().first;

    meshData->read_defined_input_fields(restartIndex);

    if (_distance_type == DistanceType::CELL)
      mesh.buildLocalElementIDs();

    mesh.setInitialStateTime(
        restartIndex > 0
            ? meshData->get_input_ioss_region()->get_state_time(restartIndex)
            : 0.0);

    delete meshData;
  }
};

template <typename T>
std::string vec2string(std::vector<T> const &s, std::string const &delim = ", ")
{
  assert(s.size() > 1);

  std::ostringstream ss;
  std::copy(s.begin(), s.end(),
            std::ostream_iterator<std::string>{ss, delim.c_str()});
  auto delimited_items = ss.str().erase(ss.str().length() - delim.size());
  return "[" + delimited_items + "]";
}

bool check_names(MPI_Comm comm, std::string const &filename,
                 std::vector<std::string> const &block_names,
                 std::vector<std::string> const &wall_names)
{
  Ioss::Init::Initializer ioss_initializer;
  Ioss::DatabaseIO *ioss_db =
      Ioss::IOFactory::create("exodus", filename, Ioss::READ_MODEL, comm);

  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);

  if (ioss_db == nullptr || !ioss_db->ok(true))
  {
    if (comm_rank == 0)
      std::cerr << "ERROR: Could not open file " << filename << "\n";
    return false;
  }

  Ioss::Region region(ioss_db, "mesh_region");

  if (!block_names.empty())
  {
    auto const &element_blocks = region.get_element_blocks();
    std::vector<std::string> element_block_names;
    for (auto const &block : element_blocks)
      element_block_names.push_back(block->name());

    for (auto const &block_name : block_names)
    {
      if (std::find(element_block_names.begin(), element_block_names.end(),
                    block_name) != element_block_names.end())
        continue;

      if (comm_rank == 0)
        std::cerr << "Element block \"" << block_name
                  << "\" not found in mesh. Available element blocks: "
                  << vec2string(element_block_names) << "\n";
      return false;
    }
  }

  auto const &sidesets = region.get_sidesets();
  std::vector<std::string> sideset_names;
  for (auto const &sideset : sidesets)
    sideset_names.push_back(sideset->name());

  for (auto const &wall_name : wall_names)
  {
    if (std::find(sideset_names.begin(), sideset_names.end(), wall_name) !=
        sideset_names.end())
      continue;

    if (comm_rank == 0)
      std::cerr << "Sideset \"" << wall_name
                << "\" not found in mesh. Available sidesets: "
                << vec2string(sideset_names) << "\n";
    return false;
  }

  return true;
}

Teuchos::RCP<panzer_stk::STK_Interface>
build_mesh(MPI_Comm comm, std::string const &filename,
           std::vector<std::string> &block_names,
           std::string const &distance_field_name, DistanceType distance_type,
           int restart_index)
{
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);

  STKMeshFactory factory(filename, distance_type, restart_index);
  auto mesh = factory.buildUncommitedMesh(comm);

  if (block_names.empty())
    mesh->getElementBlockNames(block_names);

  for (auto const &block_name : block_names)
  {
    if (distance_type == DistanceType::NODE)
      mesh->addSolutionField(distance_field_name, block_name);
    else if (distance_type == DistanceType::CELL)
      mesh->addCellField(distance_field_name, block_name);
  }

  factory.completeMeshConstruction(*mesh, comm);

  return mesh;
}

std::vector<panzer::Workset>
build_worksets(Teuchos::RCP<panzer_stk::STK_Interface> const &mesh,
               std::vector<std::string> const &block_names,
               std::string const &basis_type, int const basis_order,
               int int_order)
{
  using Teuchos::rcp;

  std::vector<panzer::Workset> worksets;
  for (auto const &block_name : block_names)
  {
    panzer::CellData cell_data(workset_size, mesh->getCellTopology(block_name));

    panzer::WorksetNeeds needs;
    auto basis = rcp(new panzer::PureBasis(basis_type, basis_order, cell_data));
    auto ir = rcp(new panzer::IntegrationRule(int_order, cell_data));
    needs.bases.push_back(basis);
    needs.int_rules.push_back(ir);
    needs.cellData = cell_data;

    auto workset_factory = Teuchos::rcp(new panzer_stk::WorksetFactory(mesh));

    panzer::WorksetContainer workset_container;
    workset_container.setFactory(workset_factory);
    workset_container.setNeeds(block_name, needs);

    panzer::WorksetDescriptor workset_descriptor(block_name);
    auto block_worksets = workset_container.getWorksets(workset_descriptor);
    worksets.insert(worksets.end(), block_worksets->begin(),
                    block_worksets->end());
  }
  return worksets;
}
