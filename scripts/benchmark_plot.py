#!/usr/bin/env python3
"""benchmark_plot.py

Usage:
  benchmark_plot.py -a ALGORITHM [-b BACKENDS...] -g GEOMETRY -i INPUT... [-n] [-e] [-t TITLE] [-o OUTPUT_FILE]
  benchmark_plot.py (-h | --help)

Options:
  -h --help                         Show this screen.
  -a ALGORITHM --algo ALGORITHM     Algorithm ['construction', 'radius_search', 'radius_callback_search', 'knn_search', 'knn_callback_search']
  -b BACKENDS --backends BACKENDS   Backends to parse [default: all]
  -g GEOMETRY --geometry GEOMETRY   Geometry (source cloud) ['filled_box', 'hollow_box']
  -i FILE --input-files=FILE        Input file(s) containing benchmark results in JSON format
  -n --numbers                      Plot numbers [default: False]
  -e --errors                       Plot error bars [default: False]
  -o FILE --output-file=FILE        Output file
  -t TITLE --title=TITLE            Plot title
"""
import matplotlib.pyplot as plt
import numpy as np
import math
import re
from docopt import docopt

from benchmark import load_json, parse_benchmark, allowed_algorithms, allowed_geometries

def find_available_backends(input_jsons):
    arborx_backend_regex = "[^<]*<ArborX::BVH<([^>]*)>.*"

    backends = set()
    for input_json in input_jsons:
        for benchmark in input_json:
            search_result = re.search(arborx_backend_regex, benchmark['name'])
            if search_result != None:
                backends.add(search_result.group(1))
    backends = list(backends)
    backends.sort()

    return backends

class NotMatchingValuesAndPredicatesSizesException(Exception):
    def __init__(self, message, i, backend):
        super().__init__(message)
        self.i = i
        self.backend = backend

def populate_data(input_jsons, backends):
    all_num_primitives  = []
    all_rates = []
    all_errors = []

    for i in range(len(backends)):
        # For every backend, find all (num_primitives, rate, error) in all input files
        backend = backends[i]

        backend_num_primitives = []
        backend_rates = []
        backend_errors = []

        implementation = 'ArborX::BVH<' + backend + '>'
        for i in range(len(input_jsons)):
            file_num_primitives, file_num_predicates, file_rates  = parse_benchmark(input_jsons[i], algorithm, implementation, 'median', geometry)
            _,                   _,                   file_errors = parse_benchmark(input_jsons[i], algorithm, implementation, 'stddev', geometry)
            if algorithm != 'construction' and file_num_primitives != file_num_predicates:
                raise NotMatchingValuesAndPredicatesSizesException("", i, backend)

            backend_num_primitives = backend_num_primitives +  file_num_primitives
            backend_rates = backend_rates + file_rates
            backend_errors = backend_errors + file_errors

        all_num_primitives.append(backend_num_primitives)
        all_rates.append(backend_rates)
        all_errors.append(backend_errors)

    num_unique_primitives = []
    for i in range(len(all_num_primitives)):
        num_unique_primitives = num_unique_primitives + all_num_primitives[i]
    num_unique_primitives = list(set(num_unique_primitives))
    num_unique_primitives.sort()

    return num_unique_primitives, all_num_primitives, all_rates, all_errors,

def backends_comparison_rate_figure(input_files, algorithm, geometry, backends, plot_numbers = False, plot_errors = True):
    # Read in files to produce JSON
    input_jsons = []
    for input_file in input_files:
        input_jsons.append(load_json(input_file))

    # Phase 0: parse backends
    backends = backends
    if backends[0] == "all":
        # Figure out a list of available backends by populating a set by
        # searching a regular expression in all input files
        backends = find_available_backends(input_jsons)

    # Phase 1: data setup
    try:
        num_unique_primitives, all_num_primitives, all_rates, all_errors = populate_data(input_jsons, backends)
    except NotMatchingValuesAndPredicatesSizesException as e:
        raise Exception('The number of predicates does not match the number of primitives for "' + e.backend + '" in "' +  input_files[e.i] + "'")
    n_unique = len(num_unique_primitives)

    # If the same data point (algorithm, backend, num_sources) is present in
    # multiple files, choose only the last one for plotting
    y_scaling = 10**6
    rates = np.zeros([len(backends), n_unique])
    errors = np.zeros([len(backends), 2, n_unique])
    for i in range(0, len(backends)):
        for j in range(0, len(all_num_primitives[i])):
            index = num_unique_primitives.index(all_num_primitives[i][j])
            rates[i, index] = all_rates[i][j]/y_scaling
            errors[i, 0, index] = 1 - all_errors[i][j]/all_rates[i][j]
            errors[i, 1, index] = 1 + all_errors[i][j]/all_rates[i][j]

    # Phase 2: plot generation
    plt.figure(figsize=(max(n_unique, 6), 5))

    ax = plt.subplot(111)
    patterns = ('/', 'x', '\\', '.', 'o', 'O')

    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = round(rect.get_height(),1)
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=18)

    # Create TeX-style label
    def primitive_label(n):
        e = math.floor(math.log(n + 1, 10))
        r = math.floor(n/10**e)
        return (str(r) + '$\cdot $' if r > 1 else '') +  '$10^' + str(e) + '$'

    width = 0.6
    scale = 1/len(backends)
    for i in range(0, len(backends)):
        barplot = ax.bar(np.arange(n_unique) + scale*(i-.5*(len(backends)-1)) * width,
                         width=scale*width*np.ones(n_unique), height=rates[i, :],
                         yerr=(errors[i, :] if plot_errors == True else None),
                         capsize=5, hatch=patterns[i % len(patterns)],
                         edgecolor='k',
                         error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2),
                         label=backends[i])

        if plot_numbers:
            autolabel(barplot)

    ax.legend(loc='upper left', fontsize=20)

    ax.set_xlabel('Number of indexed objects', fontsize=20)
    plt.xticks(fontsize=18)
    ax.set_xticks(np.arange(n_unique))
    ax.set_xticklabels([primitive_label(size) for size in num_unique_primitives])
    ax.tick_params(axis='x', which='major', bottom='off', top='off')
    ax.set_xlim([-0.5, n_unique - 0.5])

    ax.set_ylabel('Rate (million queries/second)', fontsize=20)
    plt.yticks(fontsize=18)
    ax.yaxis.grid('on')
    ax.set_ylim([0, 1.1*ax.get_ylim()[1]])

if __name__ == '__main__':

    # Process input
    options = docopt(__doc__)

    algorithm = options['--algo']
    backends = options['--backends']
    geometry = options['--geometry']
    input_files = options['--input-files']
    output_file = options['--output-file']
    plot_numbers = options['--numbers']
    plot_errors = options['--errors']
    plot_title = options['--title']

    # Check input
    if geometry not in allowed_geometries:
        raise Exception('Unknown geometry: "' + geometry + '". Allowed values are ' + str(allowed_geometries))

    if algorithm not in allowed_algorithms:
        raise Exception('Unknown algorithm: "' + algorithm + '". Allowed values are ' + str(allowed_algorithms))

    backends_comparison_rate_figure(input_files, algorithm=algorithm, geometry=geometry,
        backends=backends, plot_numbers=plot_numbers, plot_errors=plot_errors)

    if plot_title != None:
        plt.title(plot_title, fontsize=18)
    else:
        plt.title(algorithm, fontsize=18)

    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, bbox_inches='tight', dpi=600)
    else:
        plt.show()
