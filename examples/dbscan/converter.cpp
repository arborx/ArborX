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

#include <boost/program_options.hpp>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <vector>

struct Point
{
  float _data[3] = {};
  constexpr Point() noexcept = default;
  constexpr Point(float x, float y, float z)
      : _data{x, y, z}
  {
  }
};

std::vector<Point> loadHACCData(std::string const &filename)
{
  std::cout << "Assuming HACC data.\n";
  std::cout << "Reading in \"" << filename << "\" in binary mode...";
  std::cout.flush();

  std::ifstream input(filename, std::ifstream::binary);
  if (!input.good())
    throw std::runtime_error("Cannot open file");

  int num_points = 0;
  input.read(reinterpret_cast<char *>(&num_points), sizeof(int));

  std::vector<float> x(num_points);
  std::vector<float> y(num_points);
  std::vector<float> z(num_points);
  input.read(reinterpret_cast<char *>(x.data()), num_points * sizeof(float));
  input.read(reinterpret_cast<char *>(y.data()), num_points * sizeof(float));
  input.read(reinterpret_cast<char *>(z.data()), num_points * sizeof(float));
  input.close();
  std::cout << "done\nRead in " << num_points << " points" << std::endl;

  std::vector<Point> v(num_points);
  for (int i = 0; i < num_points; i++)
  {
    v[i] = {x[i], y[i], z[i]};
  }

  return v;
}

// Next Generation Simulation (NGSIM) Vehicle Trajectories data reader.
//
// NGSIM data consists of vehicle trajectory data collected by NGSIM
// researchers on three highways in Los Angeles, CA, Emeryville, CA, and
// Atlanta, GA. The trajectory data have been transcribed for every vehicle
// from the footage of video cameras using NGVIDEO.
//
// The data was used in Mustafa et al "An experimental comparison of GPU
// techniques for DBSCAN clustering", IEEE International Conference on Big Data,
// 2019.
//
// The data can be found at
// https://catalog.data.gov/dataset/next-generation-simulation-ngsim-vehicle-trajectories-and-supporting-data
// (direct link
// https://data.transportation.gov/api/views/8ect-6jqj/rows.csv?accessType=DOWNLOAD).
//
// Among other attributes, each data points has a timestamp, vehicle ID, local
// orad coordinates, global coordinates, vehicle length, width, velocity and
// acceleration.
//
// The code here is different from the source code for the Mustafa2019 paper.
// In that codebase, they seem to have a filtered file that only contains the
// global coordinates and not all the other data fields.
std::vector<Point> loadNGSIMData(std::string const &filename)
{
  std::cout << "Assuming NGSIM data.\n";
  std::cout << "Reading in \"" << filename << "\" in text mode...";
  std::cout.flush();

  std::ifstream file(filename);
  if (!file.good())
    throw std::runtime_error("Cannot open file");

  std::string thisWord, line;

  std::vector<Point> v;
  int n_points = 0;

  // ignore first line that contains the descriptions
  getline(file, thisWord);
  while (file.good())
  {
    if (!getline(file, line))
      break;

    std::stringstream ss(line);
    // GVehicle_ID,Frame_ID,Total_Frames,Global_Time,Local_X,Local_Y
    for (int i = 0; i < 6; ++i)
      getline(ss, thisWord, ',');
    // Global_X,Global_Y
    getline(ss, thisWord, ',');
    float longitude = stof(thisWord);
    getline(ss, thisWord, ',');
    float latitude = stof(thisWord);
    v.emplace_back(longitude, latitude, 0.f);
    // v_length,v_Width,v_Class,v_Vel,v_Acc,Lane_ID,O_Zone,D_Zone,Int_ID,Section_ID,Direction,Movement,Preceding,Following,Space_Headway,Time_Headway,Location
    for (int i = 0; i < 16; ++i)
      getline(ss, thisWord, ',');
    getline(ss, thisWord, ',');
    ++n_points;
  }
  std::cout << "done\nRead in " << v.size() << " points" << std::endl;
  return v;
}

// Taxi Service Trajectory Prediction Challenge data reader.
//
// The data consists of the trajectories of 442 taxis running in the city of
// Porto, Portugal, over the period of one year. This is a dataset with over
// 1,710,000+ trajectories with 81,000,000+ points in total.
//
// The data can be found at
// https://archive.ics.uci.edu/ml/datasets/Taxi+Service+Trajectory+-+Prediction+Challenge,+ECML+PKDD+2015
// (direct link
// https://archive.ics.uci.edu/ml/machine-learning-databases/00339/train.csv.zip).
//
// Every data point in this dataset has, besides longitude and latitude values,
// a unique identifier for each taxi trip, taxi ID, timestamp, and user
// information.
std::vector<Point> loadTaxiPortoData(std::string const &filename)
{
  std::cout << "Assuming TaxiPorto data.\n";
  std::cout << "Reading in \"" << filename << "\" in text mode...";
  std::cout.flush();

  FILE *fp_data = fopen(filename.c_str(), "rb");
  if (fp_data == nullptr)
    throw std::runtime_error("Cannot open file");
  char line[100000];

  // This function reads and segments trajectories in dataset in the following
  // format: The first line indicates number of variables per point (I'm
  // ignoring that and assuming 2) The second line indicates total trajectories
  // in file (I'm ignoring that and observing how many are there by reading
  // them). All lines that follow contains a trajectory separated by new line.
  // The first number in the trajectory is the number of points followed by
  // location points separated by spaces

  std::vector<float> longitudes;
  std::vector<float> latitudes;
  std::vector<Point> v;

  int lineNo = -1;
  int wordNo = 0;
  int lonlatno = 100;

  float thisWord;
  while (fgets(line, sizeof(line), fp_data))
  {
    if (lineNo > -1)
    {
      char *pch;
      char *end_str;
      wordNo = 0;
      lonlatno = 0;
      pch = strtok_r(line, "\"[", &end_str);
      while (pch != nullptr)
      {
        if (wordNo > 0)
        {
          char *pch2;
          char *end_str2;

          pch2 = strtok_r(pch, ",", &end_str2);

          if (strcmp(pch2, "]") < 0 && lonlatno < 255)
          {

            thisWord = atof(pch2);

            if (thisWord != 0.00000)
            {
              if (thisWord > -9 && thisWord < -7)
              {
                longitudes.push_back(thisWord);
                // printf("lon %f",thisWord);
                pch2 = strtok_r(nullptr, ",", &end_str2);
                thisWord = atof(pch2);
                if (thisWord < 42 && thisWord > 40)
                {
                  latitudes.push_back(thisWord);
                  // printf(" lat %f\n",thisWord);

                  lonlatno++;
                }
                else
                {
                  longitudes.pop_back();
                }
              }
            }
          }
        }
        pch = strtok_r(nullptr, "[", &end_str);
        wordNo++;
      }
      // printf("num lonlat were %d x 2\n",lonlatno);
    }
    lineNo++;
    if (lonlatno <= 0)
    {
      lineNo--;
    }

    // printf("Line %d\n",lineNo);
  }
  fclose(fp_data);

  int num_points = longitudes.size();
  assert(longitudes.size() == latitudes.size());
  v.reserve(num_points);
  for (int i = 0; i < num_points; ++i)
    v.emplace_back(longitudes[i], latitudes[i], 0.f);

  std::cout << "done\nRead in " << v.size() << " points" << std::endl;

  return v;
}

// 3D Road Network data reader.
//
// The data consists of more than 400,000 points from the road network of North
// Jutland in Denmark.
//
// The data can be found at
// https://archive.ics.uci.edu/ml/datasets/3D+Road+Network+(North+Jutland,+Denmark)
// (direct link
// https://archive.ics.uci.edu/ml/machine-learning-databases/00246/3D_spatial_network.txt).
//
// Each data point contains its ID, longitude, latitude, and altitude.
std::vector<Point> load3DRoadNetworkData(std::string const &filename)
{
  std::cout << "Assuming 3DRoadNetwork data.\n";
  std::cout << "Reading in \"" << filename << "\" in text mode...";
  std::cout.flush();

  std::ifstream file(filename);
  assert(file.good());
  if (!file.good())
    throw std::runtime_error("Cannot open file");

  std::vector<Point> v;

  std::string thisWord;
  while (file.good())
  {
    getline(file, thisWord, ',');
    getline(file, thisWord, ',');
    float longitude = stof(thisWord);
    getline(file, thisWord, ',');
    float latitude = stof(thisWord);
    v.emplace_back(longitude, latitude, 0.f);
  }
  // In Mustafa2019 they discarded the last item read but it's not quite clear
  // if/why this was necessary.
  // lon_ptr.pop_back();
  // lat_ptr.pop_back();
  std::cout << "done\nRead in " << v.size() << " points" << std::endl;

  return v;
}

// SW data reader.
//
// SW data consists of ionospheric total electron content datasets collected by
// GPS receivers.
//
// Data preprocessing described in Pankratius et al "GPS Data Processing for
// Scientific Studies of the Earth’s Atmosphere and Near-Space Environment".
// Springer International Publishing, 2015, pp. 1–12.
//
// The data was used in Gowanlock et al "Clustering Throughput Optimization on
// the GPU", IPDPS, 2017, pp. 832-841.
//
// The data is available at
// ftp://gemini.haystack.mit.edu/pub/informatics/dbscandat.zip
//
// The data file is a text file. Each line contains three floating point
// numbers separated by ','. The fields corespond to longitude, latitude, and
// total electron content (TEC). The TEC field is unused in the Gowanlock's
// paper, as according to the author:
//   because in the application scenario of monitoring space weather, we
//   typically first selected the data points based on TEC, and then cluster
//   the positions of the points
std::vector<Point> loadSWData(std::string const &filename)
{
  std::cout << "Assuming SW data.\n";
  std::cout << "Reading in \"" << filename << "\" in text mode...";
  std::cout.flush();

  std::ifstream input;
  input.open(filename);
  if (!input.good())
    throw std::runtime_error("Cannot open file");

  std::vector<Point> v;
  while (input.good())
  {
    std::string line;
    if (!std::getline(input, line))
      break;
    std::istringstream line_stream(line);

    std::string word;
    std::getline(line_stream, word, ','); // longitude field
    float longitude = std::stof(word);
    std::getline(line_stream, word, ','); // latitude field
    float latitude = std::stof(word);
    std::getline(line_stream, word, ','); // TEC field (ignored)

    v.emplace_back(longitude, latitude, 0.f);
  }
  input.close();
  std::cout << "done\nRead in " << v.size() << " points" << std::endl;

  return v;
}

// Gaia data reader.
//
// Gaia catalog (data release 2) contains 1.69 billion points
//
// Scientific Studies of the Earth’s Atmosphere and Near-Space Environment".
// Springer International Publishing, 2015, pp. 1–12.
//
// The data was used in Gowanlock "Hybrid CPU/GPU Clustering in Shared Memory
// on the Billion Point Scale", 2019
//
// The data is available at
// https://rcdata.nau.edu/gowanlock_lab/datasets/ICS19_data/gaia_dr2_ra_dec_50M.txt.
//
// The data file is a text file. Each line contains two floating point
// numbers separated by ','. The fields corespond to longitude, latitude.
std::vector<Point> loadGaiaData(std::string const &filename)
{
  std::cout << "Assuming Gaia data.\n";
  std::cout << "Reading in \"" << filename << "\" in text mode...";
  std::cout.flush();

  std::ifstream input;
  input.open(filename);
  if (!input.good())
    throw std::runtime_error("Cannot open file");

  std::vector<Point> v;
  while (input.good())
  {
    std::string line;
    if (!std::getline(input, line))
      break;
    std::istringstream line_stream(line);

    std::string word;
    std::getline(line_stream, word, ','); // longitude field
    float longitude = std::stof(word);
    std::getline(line_stream, word, ','); // latitude field
    float latitude = std::stof(word);

    v.emplace_back(longitude, latitude, 0.f);
  }
  input.close();
  std::cout << "done\nRead in " << v.size() << " points" << std::endl;

  return v;
}

std::vector<Point> loadData(std::string const &filename,
                            std::string const &reader_type)
{
  if (reader_type == "hacc")
    return loadHACCData(filename);
  else if (reader_type == "ngsim")
    return loadNGSIMData(filename);
  else if (reader_type == "taxiporto")
    return loadTaxiPortoData(filename);
  else if (reader_type == "3droad")
    return load3DRoadNetworkData(filename);
  else if (reader_type == "sw")
    return loadSWData(filename);
  else if (reader_type == "gaia")
    return loadGaiaData(filename);
  else
    throw std::runtime_error("Unknown reader type: \"" + reader_type + "\"");
}

int main(int argc, char *argv[])
{
  namespace bpo = boost::program_options;

  std::string input_file;
  std::string output_file;
  std::string reader;

  bpo::options_description desc("Allowed options");
  // clang-format off
    desc.add_options()
        ( "help", "help message" )
        ( "input", bpo::value<std::string>(&input_file), "file containing data" )
        ( "output", bpo::value<std::string>(&output_file), "file to contain the results" )
        ( "reader", bpo::value<std::string>(&reader), "reader type" )
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

  auto points = loadData(input_file, reader);

  std::ofstream out(output_file, std::ofstream::binary);
  int n = points.size();
  out.write((char *)&n, sizeof(int));
  out.write((char *)points.data(), sizeof(Point) * n);

  return EXIT_SUCCESS;
}
