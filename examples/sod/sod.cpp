/****************************************************************************
 * Copyright (c) 2017-2021 by the ArborX authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <ArborX_Ray.hpp> // Vector
#include <ArborX_SOD.hpp>
#include <ArborX_Version.hpp>

#include <Kokkos_Core.hpp>

#include <boost/program_options.hpp>

#include <fstream>

struct InputData
{
  template <typename T>
  using View = Kokkos::View<T *, Kokkos::HostSpace>;

  View<ArborX::Point> particles;
  View<float> particle_masses;

  View<ArborX::Point> fof_halo_centers;
  View<float> fof_halo_masses;

  InputData()
      : particles("particles", 0)
      , particle_masses("particle_masses", 0)
      , fof_halo_centers("fof_halo_centers", 0)
      , fof_halo_masses("fof_halo_masses", 0)
  {
  }
};

template <typename ExecutionSpace, typename Permute, typename View>
std::enable_if_t<Kokkos::is_view<View>{} && View::rank == 1>
applyPermutation(ExecutionSpace const &exec_space, Permute const &permute,
                 View view)
{
  auto const n = view.extent_int(0);
  ARBORX_ASSERT(permute.extent_int(0) == n);

  auto view_clone = ArborX::clone(exec_space, view);
  for (int i = 0; i < n; ++i)
    view(i) = view_clone(permute(i));
}

template <typename ExecutionSpace, typename Permute, typename View>
std::enable_if_t<Kokkos::is_view<View>{} && View::rank == 2>
applyPermutation2(ExecutionSpace const &exec_space, Permute const &permute,
                  View view)
{
  auto const n = view.extent_int(0);
  ARBORX_ASSERT(permute.extent_int(0) == n);

  auto const m = view.extent_int(1);

  auto view_clone = ArborX::clone(exec_space, view);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j)
      view(i, j) = view_clone(permute(i), j);
}

void loadParticlesData(std::string const &filename, InputData &in,
                       int max_num_points = -1)
{
  std::cout << "Reading in \"" << filename << "\" in binary mode...";
  std::cout.flush();

  std::ifstream input(filename, std::ifstream::binary);
  ARBORX_ASSERT(input.good());

  int N;
  input.read(reinterpret_cast<char *>(&N), sizeof(int));

  int n = N;
  if (max_num_points > 0 && max_num_points < N)
    n = max_num_points;

  Kokkos::resize(Kokkos::WithoutInitializing, in.particles, n);
  Kokkos::resize(Kokkos::WithoutInitializing, in.particle_masses, n);

  for (int d = 0; d < 3; ++d)
  {
    std::vector<float> tmp(n);
    input.read(reinterpret_cast<char *>(tmp.data()), n * sizeof(float));
    input.ignore((N - n) * sizeof(float));

    for (int i = 0; i < n; ++i)
      in.particles(i)[d] = tmp[i];
  }
  input.read(reinterpret_cast<char *>(in.particle_masses.data()),
             n * sizeof(float));
  input.ignore((N - n) * sizeof(float));

  std::cout << "done\nRead in " << n << " particles" << std::endl;

  input.close();
}

void loadHalosData(std::string const &filename, InputData &in,
                   Kokkos::View<int64_t *, Kokkos::HostSpace> &in_fof_halo_tags,
                   ArborX::SODOutputData<Kokkos::HostSpace> &out)
{
  std::cout << "Reading in \"" << filename << "\" in binary mode...";
  std::cout.flush();

  std::ifstream input(filename, std::ifstream::binary);
  ARBORX_ASSERT(input.good());

  int num_halos;
  input.read(reinterpret_cast<char *>(&num_halos), 4);

  auto read_vector = [&input](auto &v, int n) {
    v.resize(n);
    input.read(reinterpret_cast<char *>(v.data()), n * sizeof(v[0]));
  };
  auto read_view = [&input](auto &view, int n) {
    Kokkos::resize(Kokkos::WithoutInitializing, view, n);
    input.read(reinterpret_cast<char *>(view.data()),
               n * sizeof(typename std::decay_t<decltype(view)>::value_type));
  };

  Kokkos::View<int *, Kokkos::HostSpace> in_fof_halo_sizes("fof_halo_sizes", 0);
  Kokkos::View<int64_t *, Kokkos::HostSpace> out_sod_halo_sizes(
      "sod_halo_sizes", 0);

  read_view(in_fof_halo_tags, num_halos);
  read_view(in_fof_halo_sizes, num_halos);
  read_view(in.fof_halo_masses, num_halos);
  {
    std::vector<float> x, y, z;
    read_vector(x, num_halos);
    read_vector(y, num_halos);
    read_vector(z, num_halos);

    Kokkos::resize(Kokkos::WithoutInitializing, in.fof_halo_centers, num_halos);
    for (int i = 0; i < num_halos; i++)
      in.fof_halo_centers(i) = {x[i], y[i], z[i]};
  }
  read_view(out.sod_halo_masses, num_halos);
  read_view(out_sod_halo_sizes, num_halos);
  read_view(out.sod_halo_rdeltas, num_halos);

  // Filter out
  // - small halos (FOF halo size < 500)
  // - invalid halos (SOD count size = -101)
  auto swap = [](auto &view, int i, int j) { std::swap(view(i), view(j)); };
  int i = 0;
  int num_filtered = 0;
  do
  {
    if (in_fof_halo_sizes(i) < 500 || out_sod_halo_sizes(i) < 0)
    {
      // Instead of using erase(), swap with the last element
      ++num_filtered;

      int j = num_halos - num_filtered;
      if (i < j)
      {
        swap(in_fof_halo_tags, i, j);
        swap(in_fof_halo_sizes, i, j);
        swap(in.fof_halo_masses, i, j);
        swap(in.fof_halo_centers, i, j);
        swap(out.sod_halo_masses, i, j);
        swap(out_sod_halo_sizes, i, j);
        swap(out.sod_halo_rdeltas, i, j);
      }
    }
    else
    {
      ++i;
    }
  } while (i < num_halos - num_filtered);

  if (num_filtered > 0)
  {
    num_halos -= num_filtered;
    Kokkos::resize(in_fof_halo_tags, num_halos);
    Kokkos::resize(in_fof_halo_sizes, num_halos);
    Kokkos::resize(in.fof_halo_masses, num_halos);
    Kokkos::resize(in.fof_halo_centers, num_halos);
    Kokkos::resize(out.sod_halo_masses, num_halos);
    Kokkos::resize(out_sod_halo_sizes, num_halos);
    Kokkos::resize(out.sod_halo_rdeltas, num_halos);
  }

  printf("done\nRead in %d halos [%d total, %d filtered]\n", num_halos,
         num_halos + num_filtered, num_filtered);

  // Sort halos by tags for consistency
  auto host_space = Kokkos::DefaultHostExecutionSpace{};
  auto permute = ArborX::Details::sortObjects(host_space, in_fof_halo_tags);
  applyPermutation(host_space, permute, in.fof_halo_masses);
  applyPermutation(host_space, permute, in.fof_halo_centers);
  applyPermutation(host_space, permute, out.sod_halo_masses);
  applyPermutation(host_space, permute, out.sod_halo_rdeltas);

  input.close();
}

void loadProfilesData(
    std::string const &filename, ArborX::SODOutputData<Kokkos::HostSpace> &out,
    Kokkos::View<int64_t *, Kokkos::HostSpace> &out_fof_halo_tags)
{
  std::cout << "Reading in \"" << filename << "\" in binary mode...";
  std::cout.flush();

  std::ifstream input(filename, std::ifstream::binary);
  ARBORX_ASSERT(input.good());

  // The profile file does not contain first bin with < R_min)
  ARBORX_ASSERT(NUM_SOD_BINS == 21);

  int num_records;
  input.read(reinterpret_cast<char *>(&num_records), 4);
  ARBORX_ASSERT(num_records % (NUM_SOD_BINS - 1) == 0);

  int num_halos = num_records / (NUM_SOD_BINS - 1);

  auto read_view = [&input](auto &view, int n) {
    Kokkos::resize(Kokkos::WithoutInitializing, view, n);
    input.read(reinterpret_cast<char *>(view.data()),
               n * sizeof(typename std::decay_t<decltype(view)>::value_type));
  };
  auto read_bin_view = [&input](auto &view, int n) {
    using view_type = std::decay_t<decltype(view)>;
    using value_type = typename view_type::value_type;

    Kokkos::View<value_type *, typename view_type::device_type> v(
        Kokkos::ViewAllocateWithoutInitializing("tmp"), n * (NUM_SOD_BINS - 1));
    input.read(reinterpret_cast<char *>(v.data()),
               v.extent(0) * sizeof(value_type));

    // First bin is unused, shift data
    Kokkos::resize(Kokkos::WithoutInitializing, view, n);
    for (int i = 0; i < n; ++i)
    {
      view(i, 0) = -1; // just want a value that is clearly unidentifiable
      for (int j = 0; j < NUM_SOD_BINS - 1; ++j)
        view(i, j + 1) = v(i * (NUM_SOD_BINS - 1) + j);
    }
  };

  // FOF halo tags are repeated in groups of size NUM_SOD_BINS-1, make them
  // unique
  read_view(out_fof_halo_tags, num_records);
  for (int i = 1; i < num_halos; ++i)
    out_fof_halo_tags(i) = out_fof_halo_tags(i * (NUM_SOD_BINS - 1));
  Kokkos::resize(out_fof_halo_tags, num_halos);

  read_bin_view(out.sod_halo_bin_ids, num_halos);
  read_bin_view(out.sod_halo_bin_counts, num_halos);
  read_bin_view(out.sod_halo_bin_masses, num_halos);
  read_bin_view(out.sod_halo_bin_outer_radii, num_halos);
  read_bin_view(out.sod_halo_bin_rhos, num_halos);
  read_bin_view(out.sod_halo_bin_rho_ratios, num_halos);
  read_bin_view(out.sod_halo_bin_radial_velocities, num_halos);

  // Sort halos by tags for consistency
  auto host_space = Kokkos::DefaultHostExecutionSpace{};
  auto permute = ArborX::Details::sortObjects(host_space, out_fof_halo_tags);
  applyPermutation2(host_space, permute, out.sod_halo_bin_ids);
  applyPermutation2(host_space, permute, out.sod_halo_bin_counts);
  applyPermutation2(host_space, permute, out.sod_halo_bin_masses);
  applyPermutation2(host_space, permute, out.sod_halo_bin_outer_radii);
  applyPermutation2(host_space, permute, out.sod_halo_bin_rhos);
  applyPermutation2(host_space, permute, out.sod_halo_bin_rho_ratios);
  applyPermutation2(host_space, permute, out.sod_halo_bin_radial_velocities);

  printf("done\nRead in %d halos\n", num_halos);

  input.close();
}

int main(int argc, char *argv[])
{
  using ExecutionSpace = Kokkos::DefaultExecutionSpace;

  Kokkos::ScopeGuard guard(argc, argv);

  std::cout << "ArborX version    : " << ArborX::version() << std::endl;
  std::cout << "ArborX hash       : " << ArborX::gitCommitHash() << std::endl;

  namespace bpo = boost::program_options;

  int max_num_points;
  std::string filename_particles;
  std::string filename_halos;
  std::string filename_profiles;
  bool validate;

  bpo::options_description desc("Allowed options");
  // clang-format off
  desc.add_options()
      ( "help", "help message" )
      ( "filename-particles", bpo::value<std::string>(&filename_particles), "filename containing particles data" )
      ( "filename-halos", bpo::value<std::string>(&filename_halos), "filename containing halos data" )
      ( "filename-profiles", bpo::value<std::string>(&filename_profiles), "filename containing profiles data" )
      ( "max-num-points", bpo::value<int>(&max_num_points)->default_value(-1), "max number of points to read in" )
      ( "validate", bpo::bool_switch(&validate)->default_value(false), "output validation results" )
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

  // print out the runtime parameters
  printf("filename [particles] : %s [max_pts = %d]\n",
         filename_particles.c_str(), max_num_points);
  printf("filename [halos]     : %s\n", filename_halos.c_str());
  printf("filename [profiles]  : %s\n", filename_profiles.c_str());

  // read in data
  InputData input_data;
  ArborX::SODOutputData<Kokkos::HostSpace> validation_data;
  Kokkos::View<int64_t *, Kokkos::HostSpace> in_fof_halo_tags(
      "in_fof_halo_tags", 0);
  Kokkos::View<int64_t *, Kokkos::HostSpace> validation_fof_halo_tags(
      "in_fof_halo_tags", 0);
  loadParticlesData(filename_particles, input_data, max_num_points);
  loadHalosData(filename_halos, input_data, in_fof_halo_tags, validation_data);
  loadProfilesData(filename_profiles, validation_data,
                   validation_fof_halo_tags);

  int const num_halos = input_data.fof_halo_centers.extent_int(0);

  // Validate tags
  ARBORX_ASSERT(validation_fof_halo_tags.extent_int(0) ==
                in_fof_halo_tags.extent_int(0));
  for (int i = 0; i < num_halos; ++i)
    ARBORX_ASSERT(in_fof_halo_tags(i) == validation_fof_halo_tags(i));

  // run SOD
  auto output_data = ArborX::sod(
      ExecutionSpace{}, input_data.particles, input_data.particle_masses,
      input_data.fof_halo_centers, input_data.fof_halo_masses);

  // validate
  if (validate)
  {
    auto const num_halos = input_data.fof_halo_centers.extent_int(0);

    auto relative_error = [](auto a, auto b) {
      if (a != 0)
        return std::abs(float(a - b) / a);
      if (a == 0 && b != 0)
        return std::numeric_limits<float>::infinity();
      return 0.f;
    };

    float max_error;

    // outer radii
    printf(">>> validating radii\n");
    max_error = 0.f;
    for (int i = 0; i < num_halos; ++i)
    {
      bool matched = true;
      for (int bin_id = 1; bin_id < NUM_SOD_BINS; ++bin_id)
        matched &= (output_data.sod_halo_bin_outer_radii(i, bin_id) ==
                    validation_data.sod_halo_bin_outer_radii(i, bin_id));
      if (!matched)
      {
        printf("radii for halo tag %ld do not match: relative errors [",
               in_fof_halo_tags(i));
        for (int bin_id = 1; bin_id < NUM_SOD_BINS; ++bin_id)
        {
          auto error = relative_error(
              output_data.sod_halo_bin_outer_radii(i, bin_id),
              validation_data.sod_halo_bin_outer_radii(i, bin_id));
          max_error = std::max(error, max_error);
          printf(" %e", error);
        }
        printf(" ]\n");
      }
    }
    printf(">>> radii max error = %e\n", max_error);

    // bin counts
    printf(">>> validating bin counts\n");
    for (int i = 0; i < num_halos; ++i)
    {
      bool matched = true;
      for (int bin_id = 1; bin_id < NUM_SOD_BINS; ++bin_id)
        matched &= (output_data.sod_halo_bin_counts(i, bin_id) ==
                    validation_data.sod_halo_bin_counts(i, bin_id));
      if (!matched)
      {
        printf("counts for halo tag %ld do not match: validation = [",
               in_fof_halo_tags(i));
        for (int bin_id = 1; bin_id < NUM_SOD_BINS; ++bin_id)
          printf(" %d", validation_data.sod_halo_bin_counts(i, bin_id));
        printf(" ], errors = [");
        for (int bin_id = 1; bin_id < NUM_SOD_BINS; ++bin_id)
          printf(" %d", output_data.sod_halo_bin_counts(i, bin_id) -
                            validation_data.sod_halo_bin_counts(i, bin_id));
        printf(" ]\n");
      }
    }

    // bin masses
    printf(">>> validating bin masses\n");
    for (int i = 0; i < num_halos; ++i)
    {
      bool matched = true;
      for (int bin_id = 1; bin_id < NUM_SOD_BINS; ++bin_id)
        matched &= (output_data.sod_halo_bin_masses(i, bin_id) ==
                    validation_data.sod_halo_bin_masses(i, bin_id));
      if (!matched)
      {
        printf("masses for halo tag %ld do not match: relative errors [",
               in_fof_halo_tags(i));
        for (int bin_id = 1; bin_id < NUM_SOD_BINS; ++bin_id)
          printf(" %e", relative_error(
                            output_data.sod_halo_bin_masses(i, bin_id),
                            validation_data.sod_halo_bin_masses(i, bin_id)));
        printf(" ]\n");
      }
    }

    // r_delta
    printf(">>> validating rdelta\n");
    max_error = 0.f;
    for (int i = 0; i < num_halos; ++i)
    {
      float a = output_data.sod_halo_rdeltas(i);
      float b = validation_data.sod_halo_rdeltas(i);

      if (a != b)
      {
        auto error = relative_error(a, b);
        max_error = std::max(error, max_error);
        printf("%d rdelta for halo tag %ld do not match: "
               "output = %f, validation = %f, relative error = %e\n",
               i, in_fof_halo_tags(i), a, b, error);
      }
    }
    printf(">>> rdelta max error = %e\n", max_error);

    // rho
    printf(">>> validating rho\n");
    max_error = 0.f;
    for (int i = 0; i < num_halos; ++i)
    {
      bool matched = true;
      for (int bin_id = 1; bin_id < NUM_SOD_BINS; ++bin_id)
        matched &= (output_data.sod_halo_bin_rhos(i, bin_id) ==
                    validation_data.sod_halo_bin_rhos(i, bin_id));
      if (!matched)
      {
        printf("rho for halo tag %ld do not match: relative errors [",
               in_fof_halo_tags(i));
        for (int bin_id = 1; bin_id < NUM_SOD_BINS; ++bin_id)
        {
          auto error =
              relative_error(output_data.sod_halo_bin_rhos(i, bin_id),
                             validation_data.sod_halo_bin_rhos(i, bin_id));
          max_error = std::max(error, max_error);
          printf(" %e", error);
        }
        printf(" ]\n");
      }
    }
    printf(">>> rho max error = %e\n", max_error);
  }

  return EXIT_SUCCESS;
}
