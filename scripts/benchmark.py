#!/usr/bin/env python

import json
import re

def load_json(filename):
    with open(filename) as f:
        try:
            # Try loading a file using JSON reader
            data = json.loads(f.read())['benchmarks']
        except ValueError:
            # Try loading a file using ASCII reader
            data = []
            f.seek(0)   # reset the file as it was messed up above
            try:
                for line in f:
                    # Try to parse only the lines starting with "BM_"
                    if re.search('^BM_', line) != None:
                        for timer in ['manual_time', 'manual_time_mean', 'manual_time_median', 'manual_time_stddev']:
                            m = re.search('(^.*/' + timer + ') .*rate=([0-9.]*)(.*)$', line)
                            if m != None:
                                name = m.group(1)
                                rate = float(m.group(2))
                                rate_scale = m.group(3)
                                if rate_scale == 'k/s':
                                    rate = rate * 1000
                                elif rate_scale == 'M/s':
                                    rate = rate * 1000000
                                else:
                                    raise Exception('Unknown rate scaling: "' + rate_scale + '"')
                                data.append({'name' : name, 'rate' : rate})
                                break

            except RuntimeError as e:
                raise RuntimeError("Caught an error while parsing on line " + str(lineno) + ": " + e.args[0])

    return data

allowed_geometries = ['filled_box', 'hollow_box']
allowed_algorithms = ['construction', 'radius_search', 'radius_callback_search', 'knn_search', 'knn_callback_search']

# Geometry is only checked for the source cloud
# We do not check target cloud geometry as they come in pairs:
#   filled_box/filled_sphere or hollow_box/hollow_sphere
def parse_benchmark(json_data, algorithm, implementation, timer_aggregate, geometry, num_radius_passes = 2):
    if geometry not in allowed_geometries:
        raise Exception('Unknown geometry: "' + geometry + '"')
    if algorithm not in allowed_algorithms:
        raise Exception('Unknown algorithm: "' + algorithm + '"')

    geometries = { 'filled_box' : '0', 'hollow_box' : '1', 'filled_sphere' : '2', 'hollow_sphere' : '3' }

    # Escape () in implementation string
    implementation = implementation.translate(str.maketrans({"(" : "\(", ")" : "\)", "%" : "\%" }))

    sorted = 1

    # Templates:
    # BM_construction<ArborX::BVH<_DeviceType_>>/_num_primitives_/_source_point_cloud_type_
    construction_template = algorithm + '<.*' + implementation + '[^/]*/([^/]*)/' + geometries[geometry] + '/'
    # BM_knn_search<ArborX::BVH<_DeviceType_>>/_num_primitives_/_num_predicates_/_n_neighbors_/_sort_predicate_/_source_point_cloud_type_/_target_point_cloud_type_
    knn_template = algorithm + '<.*' + implementation + '[^/]*/([^/]*)/([^/]*)/[^/]*/' + str(sorted) + '/' + geometries[geometry] + '/'
    # BM_knn_search<ArborX::BVH<_DeviceType_>>/_num_primitives_/_num_predicates_/_n_neighbors_/_sort_predicates_/_buffer_size_/_source_point_cloud_type_/_target_point_cloud_type_
    # we consider (2P) to be represented by buffer_size = 0
    radius_template_1P = algorithm + '<.*' + implementation + '[^/]*/([^/]*)/([^/]*)/[^/]*/' + str(sorted) + '/[^0][^/]+/' + geometries[geometry] + '/'
    radius_template_2P = algorithm + '<.*' + implementation + '[^/]*/([^/]*)/([^/]*)/[^/]*/' + str(sorted) + '/0/' + geometries[geometry] + '/'

    num_primitives = []
    num_predicates = []
    rates    = []

    for benchmark in json_data:
        name = benchmark['name']

        # For Google Benchmark v1.5, one could do
        #  if benchmark['run_type'] != 'aggregate' or benchmark['aggregate_name'] != timer_aggregate:
            #  continue
        if re.search(timer_aggregate, name) == None:
            continue

        if algorithm == 'construction' and re.search(algorithm, name) != None:
            m = re.search(construction_template, name)
            if m != None:
                num_primitives.append(int(m.group(1)))
                rates.append(benchmark['rate'])

        elif (algorithm == 'knn_search' or algorithm == 'knn_callback_search') and re.search(algorithm, name) != None:
            m = re.search(knn_template, name)
            if m != None:
                num_primitives.append(int(m.group(1)))
                num_predicates.append(int(m.group(2)))
                rates.append(benchmark['rate'])

        elif (algorithm == 'radius_search' or algorithm == 'radius_callback_search') and re.search(algorithm, name) != None:
            m = re.search(radius_template_2P if num_radius_passes == 2 else radius_template_1P, name)
            if m != None:
                num_primitives.append(int(m.group(1)))
                num_predicates.append(int(m.group(2)))
                rates.append(benchmark['rate'])

    return num_primitives, num_predicates, rates
