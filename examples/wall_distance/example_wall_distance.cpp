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

#include "../../benchmarks/utils/ArborXBenchmark_TimeMonitor.hpp"
#include "ArborX_WallDistance.hpp"
#include <ArborX_Version.hpp>

#include <boost/program_options.hpp>

#include <string>
#include <vector>

#include <Panzer_IntegrationRule.hpp>
#include <Panzer_STK_ExodusReaderFactory.hpp>
#include <Panzer_STK_WorksetFactory.hpp>
#include <Panzer_WorksetContainer.hpp>
#include <Panzer_WorksetNeeds.hpp>
#include <Teuchos_RCPStdSharedPtrConversions.hpp>
#include <mpi.h>

constexpr int workset_size = 64;

Teuchos::RCP<panzer_stk::STK_Interface>
build_mesh(std::string const &filename, MPI_Comm comm, std::string &block_name,
           std::string const &distance_field_name,
           std::string const &distance_type)
{
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);

  panzer_stk::STK_ExodusReaderFactory factory(filename);
  auto mesh = factory.buildUncommitedMesh(comm);

  if (block_name == "")
  {
    std::vector<std::string> block_names;
    mesh->getElementBlockNames(block_names);
    KOKKOS_ASSERT(!block_names.empty());
    block_name = block_names[0];
    if (comm_rank == 0)
      std::cout << "No block name provided, using first available name: \""
                << block_name << "\"" << std::endl;
  }

  if (distance_type == "node")
    mesh->addSolutionField(distance_field_name, block_name);
  else if (distance_type == "cell")
    mesh->addCellField(distance_field_name, block_name);

  factory.completeMeshConstruction(*mesh, comm);

  return mesh;
}

auto build_worksets(Teuchos::RCP<panzer_stk::STK_Interface> const &mesh,
                    std::string const &block_name,
                    std::string const &basis_type, int const basis_order,
                    int int_order)
{
  using Teuchos::rcp;

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
  return workset_container.getWorksets(workset_descriptor);
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);

  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = typename ExecutionSpace::memory_space;

  MPI_Comm const comm = MPI_COMM_WORLD;
  int comm_rank;
  MPI_Comm_rank(comm, &comm_rank);
  int comm_size;
  MPI_Comm_size(comm, &comm_size);
  if (comm_rank == 0)
  {
    std::cout << "ArborX version    : " << ArborX::version() << std::endl;
    std::cout << "ArborX hash       : " << ArborX::gitCommitHash() << std::endl;
    std::cout << "Kokkos version    : " << ArborX::Details::KokkosExt::version()
              << std::endl;
    std::cout << "#MPI ranks        : " << comm_size << std::endl;
  }

  using Coordinate = double;

  namespace bpo = boost::program_options;

  // Strip "--help" and "--kokkos-help" from the flags passed to Kokkos if we
  // are not on MPI rank 0 to prevent Kokkos from printing the help message
  // multiply.
  auto *help_it = std::find_if(argv, argv + argc, [](std::string const &x) {
    return x == "--help" || x == "--kokkos-help";
  });
  bool is_help_present = (help_it != argv + argc);
  if (is_help_present && comm_rank != 0)
  {
    std::swap(*help_it, *(argv + argc - 1));
    --argc;
  }

  Kokkos::ScopeGuard guard(argc, argv);

  std::string basis_type;
  std::string filename;
  std::string out_filename;
  int basis_order;
  int int_order;
  std::string block_name;
  std::vector<std::string> wall_names;
  std::string distance_type;
  bool verbose;

  bpo::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
    ("help", "help message" )
    ("basis-order", bpo::value<int>(&basis_order)->default_value(1), "basis order")
    ("basis-distance_type", bpo::value<std::string>(&basis_type)->default_value("HGrad"), "basis distance_type")
    ("block-name", bpo::value<std::string>(&block_name)->default_value(""), "block name")
    ("filename", bpo::value<std::string>(&filename)->default_value("mesh.exo"), "mesh filename")
    ("int-order", bpo::value<int>(&int_order)->default_value(2), "integration order")
    ("output-filename", bpo::value<std::string>(&out_filename)->default_value("output.exo"), "output filename")
    ("type", bpo::value<std::string>(&distance_type)->default_value("node"), "type of field to write (node or cell)")
    ("verbose", bpo::bool_switch(&verbose), "verbose")
    ("wall-names", bpo::value<std::vector<std::string>>(&wall_names)->multitoken(), "names of walls")
    ;
  // clang-format on
  bpo::variables_map vm;
  bpo::store(bpo::command_line_parser(argc, argv).options(desc).run(), vm);
  bpo::notify(vm);

  if (is_help_present)
  {
    if (comm_rank == 0)
      std::cout << desc << '\n';
    MPI_Finalize();
    return 0;
  }

  if (wall_names.empty())
  {
    if (comm_rank == 0)
      std::cerr << "At least one wall name must be provided\n";
    MPI_Finalize();
    return 1;
  }

  if (distance_type != "node" && distance_type != "cell")
  {
    if (comm_rank == 0)
      std::cerr << "Invalid distance_type: " << distance_type
                << ". Must be \"node\" or \"cell\".\n";
    MPI_Finalize();
    return 2;
  }

  auto vec2string = [](std::vector<std::string> const &names) {
    if (names.empty())
      return std::string("(none)");
    std::string result = names[0];
    for (size_t i = 1; i < names.size(); ++i)
      result += ", " + names[i];
    return result;
  };

  // Print out the runtime parameters
  if (comm_rank == 0)
  {
    printf("basis order       : %d\n", basis_order);
    printf("basis type        : %s\n", basis_type.c_str());
    printf("block name        : %s\n", block_name.c_str());
    printf("filename          : %s\n", filename.c_str());
    printf("integration order : %d\n", int_order);
    printf("distance type     : %s\n", distance_type.c_str());
    printf("verbose           : %s\n", (verbose ? "true" : "false"));
    printf("wall names        : %s\n", vec2string(wall_names).c_str());
  }

  constexpr bool ReplicateSides = true;
  std::string const distance_field_name = "wall_distance";

  {
    ExecutionSpace space;

    ArborXBenchmark::TimeMonitor time_monitor;

    // Note: when running in parallel, use
    //   export IOSS_PROPERTIES="DECOMPOSITION_METHOD=RIB"
    // for automatic mesh decomposition. This requires NetCDF-C compiled
    // with parallel support.
    auto mesh = build_mesh(filename, comm, block_name, distance_field_name,
                           distance_type);

    using ArborX::Experimental::WallDistance;
    using Index2D = WallDistance<MemorySpace, 2, Coordinate, ReplicateSides>;
    using Index3D = WallDistance<MemorySpace, 3, Coordinate, ReplicateSides>;

    auto construction_time = time_monitor.getNewTimer("construction");
    construction_time->start();
    std::variant<Index2D, Index3D> wall_distance;
    if (mesh->getDimension() == 2)
      wall_distance.emplace<Index2D>(space, *mesh, wall_names);
    else
      wall_distance.emplace<Index3D>(space, *mesh, wall_names);
    space.fence();
    construction_time->stop();

    if (distance_type == "node")
    {
    }
    else if (distance_type == "cell")
    {
      auto worksets =
          build_worksets(mesh, block_name, basis_type, basis_order, int_order);

      panzer::CellData cell_data(workset_size,
                                 mesh->getCellTopology(block_name));
      panzer::IntegrationRule ir(int_order, cell_data);

      int const num_worksets = (*worksets).size();
      int const max_num_cells_per_workset = ir.dl_scalar->extent(0);
      int const num_int_points_per_cell = ir.dl_scalar->extent(1);

      Kokkos::View<Coordinate ***, MemorySpace> workset_distances(
          Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                             "Example::workset_distances"),
          num_worksets, max_num_cells_per_workset, num_int_points_per_cell);

      space.fence();

      auto query_time = time_monitor.getNewTimer("query");
      query_time->start();
      std::visit(
          [&](auto &&index) {
            index.distance(space, *worksets, ir, workset_distances);
          },
          wall_distance);
      space.fence();
      query_time->stop();

      // Copy workset distances to a flat array
      Kokkos::View<Coordinate *, MemorySpace> wall_distances(
          Kokkos::view_alloc(space, Kokkos::WithoutInitializing,
                             "Example::wall_distances"),
          num_worksets * max_num_cells_per_workset);
      for (int workset_id = 0; workset_id < num_worksets; ++workset_id)
      {
        auto const &workset = (*worksets)[workset_id];
        auto num_cells = workset.num_cells;
        if (num_cells == 0)
          continue;

        auto local_cell_ids = workset.getLocalCellIDs();
        Kokkos::parallel_for(
            "Example::copy_distances", Kokkos::RangePolicy(space, 0, num_cells),
            KOKKOS_LAMBDA(int i) {
              Coordinate avg = 0;
              for (int p = 0; p < num_int_points_per_cell; ++p)
                avg += workset_distances(workset_id, i, p);
              wall_distances(local_cell_ids(i)) = avg / num_int_points_per_cell;
            });
      }
      space.fence();

      auto wall_distances_host = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace{}, wall_distances);

      // Store wall distances in the output file
      auto *field = mesh->getCellField(distance_field_name, block_name);
      std::vector<stk::mesh::Entity> elements;
      mesh->getMyElements(block_name, elements);
      for (std::size_t el = 0; el < elements.size(); el++)
      {
        std::size_t localId = mesh->elementLocalId(elements[el]);
        double *field_data = stk::mesh::field_data(*field, elements[el]);
        KOKKOS_ASSERT(field_data != nullptr); // sanity check
        field_data[0] = wall_distances_host(localId);
      }
    }
    time_monitor.summarize(comm);

    mesh->writeToExodus(out_filename);
  }

  MPI_Finalize();

  return 0;
}
