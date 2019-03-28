#!/usr/bin/env python3
"""arborx_query_sort.py

Usage:
  arborx_query_sort.py -a ALGO -p PREFIX [-d] [-o OUTPUT_FILE]
  arborx_query_sort.py (-h | --help)

Options:
  -h --help                   Show this screen.
  -a ALGO --algo=ALGO         Query order ['untouched', 'shuffled', 'sorted']
  -d --display                Display mode
  -o FILE --output-file=FILE  Output file [default: plot.png]
  -p PREFIX --prefix=PREFIX   Input file prefix
"""
from docopt import docopt
import subprocess
import numpy
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def read_traversal(filename):
    nearest_traversal = []
    replace = {
        '[internal]': '[style=filled, color=black]',
        '[leaf]': '[style=filled, color=white]',
        '[result]': '[style=filled, color=black]',
    }
    with open(filename) as fin:
        def is_commented(line):
            return line.lstrip(' ').startswith('//')

        def is_edge(line):
            return line.count('->') > 0

        for line in fin:
            for old, new in replace.items():
                line = line.replace(old, new)
            line = line.rstrip('\n')
            if line and not is_commented(line) and not is_edge(line):
                nearest_traversal.append(line)
    return nearest_traversal


def read_tree(filename):
    tree = []
    replace = {
        '[internal]': '[style=filled, color=red]',
        '[leaf]': '[style=filled, color=green]',
        '[edge]': '',
        '[pendant]': '',
    }
    with open(filename) as fin:
        for line in fin:
            line = line.rstrip('\n')
            for old, new in replace.items():
                line = line.replace(old, new)
            tree.append(line)
    tree.insert(0, 'digraph g {')
    tree.insert(1, '    root = i0;')
    tree.append('}')
    return tree


def append_traversal_to_tree(tree, traversal):
    for line in traversal:
        tree.insert(-1, line)
    return tree


if __name__ == '__main__':
    # Process input
    options = docopt(__doc__)

    algo = options['--algo']
    display = options['--display']
    prefix = options['--prefix']
    output_file = options['--output-file']

    matrix = {}

    n = int(subprocess.check_output('ls -1 ' + prefix + '*' + algo + '* '
                                    '| wc -l', shell=True))
    assert n > 0, 'Could not find any matching files'

    matrix = numpy.zeros((n, n-1))
    for i in range(n):
        with open(prefix + '%s_%s_nearest_traversal.dot.m4'
                  % (algo, i)) as fin:
            def is_commented(line):
                return line.lstrip(' ').startswith('//')

            def is_edge(line):
                return line.count('->') > 0

            count = 0
            for line in fin:
                if line and not is_commented(line) and not is_edge(line):
                    count += 1
                    line = line.lstrip(' ')
                    entry = int(line[1:line.find(' ')])
                    if line[0] is 'i':
                        matrix[i, entry] = 1
                    elif line[0] is 'l':
                        # Uncomment for full matrix with leaves
                        # matrix[i, n-1 + entry] = 1
                        continue
                    else:
                        raise RuntimeError()

    cmap = ListedColormap(['w', 'r', 'k'])

    plt.matshow(matrix, cmap=cmap)
    plt.tight_layout()

    if display:
        plt.show()
    else:
        if output_file == "":
            output_file = algo + '.png'
        plt.savefig(output_file, bbox_inches='tight')
