/****************************************************************************
 * Copyright (c) 2012-2020 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <ArborX_HaloFinder.hpp>
#include <ArborX_Version.hpp>

#include <Kokkos_Core.hpp>

#include <boost/program_options.hpp>

#include <fstream>

std::vector<ArborX::Point> parsePoints(std::string const &filename,
                                       bool binary = false)
{
  std::cout << "Reading in \"" << filename << "\" in "
            << (binary ? "binary" : "text") << " mode...";
  std::cout.flush();

  std::ifstream input;
  if (!binary)
    input.open(filename);
  else
    input.open(filename, std::ifstream::binary);
  ARBORX_ASSERT(input.good());

  int num_points = 0;
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> z;
  if (!binary)
  {
    input >> num_points;

    x.reserve(num_points);
    y.reserve(num_points);
    z.reserve(num_points);

    auto read_float = [&input]() {
      return *(std::istream_iterator<float>(input));
    };
    std::generate_n(std::back_inserter(x), num_points, read_float);
    std::generate_n(std::back_inserter(y), num_points, read_float);
    std::generate_n(std::back_inserter(z), num_points, read_float);
  }
  else
  {
    input.read(reinterpret_cast<char *>(&num_points), sizeof(int));

    x.resize(num_points);
    y.resize(num_points);
    z.resize(num_points);
    input.read(reinterpret_cast<char *>(x.data()), num_points * sizeof(float));
    input.read(reinterpret_cast<char *>(y.data()), num_points * sizeof(float));
    input.read(reinterpret_cast<char *>(z.data()), num_points * sizeof(float));
  }
  input.close();
  std::cout << "done\nRead in " << num_points << " points" << std::endl;

  std::vector<ArborX::Point> v(num_points);
  for (int i = 0; i < num_points; i++)
  {
    v[i] = {x[i], y[i], z[i]};
  }

  return v;
}

template <typename... P, typename T>
auto vec2view(std::vector<T> const &in, std::string const &label = "")
{
  Kokkos::View<T *, P...> out(
      Kokkos::view_alloc(label, Kokkos::WithoutInitializing), in.size());
  Kokkos::deep_copy(out, Kokkos::View<T const *, Kokkos::HostSpace,
                                      Kokkos::MemoryTraits<Kokkos::Unmanaged>>{
                             in.data(), in.size()});
  return out;
}

int main(int argc, char *argv[])
{
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;
  using MemorySpace = typename ExecutionSpace::memory_space;

  Kokkos::ScopeGuard guard(argc, argv);

  namespace bpo = boost::program_options;

  std::string filename;
  bool binary;
  bool verify;
  bool print_halo_timers;
  bool print_sizes_centers;
  float linking_length;
  int min_size;

  bpo::options_description desc("Allowed options");
  // clang-format off
    desc.add_options()
        ( "help", "help message" )
        ( "filename", bpo::value<std::string>(&filename), "filename containing data" )
        ( "binary", bpo::bool_switch(&binary)->default_value(false), "binary file indicator")
        ( "linking-length", bpo::value<float>(&linking_length), "linking length (radius)" )
        ( "min-size", bpo::value<int>(&min_size)->default_value(2), "minimum halo size")
        ( "verify", bpo::bool_switch(&verify)->default_value(false), "verify connected components")
        ( "print-halo-timers", bpo::bool_switch(&print_halo_timers)->default_value(false), "print halo timers")
        ( "output-sizes-and-centers", bpo::bool_switch(&print_sizes_centers)->default_value(false), "print halo sizes and centers")
        ;
  // clang-format on
  bpo::variables_map vm;
  bpo::store(bpo::command_line_parser(argc, argv).options(desc).run(), vm);
  bpo::notify(vm);

  if (vm.count("help") > 0)
  {
    std::cout << desc << '\n';
    return 1;
  }

  std::cout << "ArborX version: " << ArborX::version() << std::endl;
  std::cout << "ArborX hash   : " << ArborX::gitCommitHash() << std::endl;

  // read in data
  auto const primitives =
      vec2view<MemorySpace>(parsePoints(filename, binary), "primitives");

  ExecutionSpace exec_space;

  Kokkos::View<int *, MemorySpace> halos_indices("halos_indices", 0);
  Kokkos::View<int *, MemorySpace> halos_offset("halos_offset", 0);
  ArborX::HaloFinder::findHalos(exec_space, primitives, halos_indices,
                                halos_offset, linking_length, min_size,
                                print_halo_timers, verify);

  if (print_sizes_centers)
  {
    auto const num_halos = static_cast<int>(halos_offset.size()) - 1;

    Kokkos::View<ArborX::Point *, MemorySpace> halos_centers(
        Kokkos::ViewAllocateWithoutInitializing("centers"), num_halos);
    Kokkos::parallel_for(
        "compute centers",
        Kokkos::RangePolicy<ExecutionSpace>(exec_space, 0, num_halos),
        KOKKOS_LAMBDA(int const i) {
          int halo_size = halos_offset(i + 1) - halos_offset(i);

          ArborX::Point halo_center{0.f, 0.f, 0.f};
          for (int j = halos_offset(i); j < halos_offset(i + 1); j++)
          {
            auto const &halo_point = primitives(halos_indices(j));
            // NOTE The explicit casts below are intended to silent warnings
            // about narrowing conversion from 'int' to 'float'.  The potential
            // issue is that 'float' can represent all integer values in the
            // range [-2^23, 2^23] but 'int' can actually represent values in
            // the range [-2^31, 2^31-1].
            halo_center[0] += halo_point[0] / (float)halo_size;
            halo_center[1] += halo_point[1] / (float)halo_size;
            halo_center[2] += halo_point[2] / (float)halo_size;
          }
          halos_centers(i) = halo_center;
        });

    auto halos_offset_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, halos_offset);
    auto halos_centers_host =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, halos_centers);
    for (int i = 0; i < num_halos; i++)
    {
      int halo_size = halos_offset_host(i + 1) - halos_offset_host(i);

      // This is HACC specific filtering
      auto const &halo_center = halos_centers_host(i);
      if (halo_center[0] >= 0 && halo_center[1] >= 0 && halo_center[2] >= 0 &&
          halo_center[0] < 64 && halo_center[1] < 64 && halo_center[2] < 64)
      {
        printf("%d %e %e %e\n", halo_size, halo_center[0], halo_center[1],
               halo_center[2]);
      }
    }
  }

  return EXIT_SUCCESS;
}
