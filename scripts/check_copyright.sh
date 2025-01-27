#!/usr/bin/env bash

year=2025

license="/****************************************************************************
 * Copyright (c) ${year}, ArborX authors                                       *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the ArborX library. ArborX is                       *
 * distributed under a BSD 3-clause license. For the licensing terms see    *
 * the LICENSE file in the top-level directory.                             *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/"
header_length=$(echo "$license" | wc -l)

temp_header_file="scripts/.temp_header"
master="origin/master"

n_wrong_licenses=0
for file in $(git diff --diff-filter=AMRU --name-only ${master}... | grep -e '.*\.\(cc\|cpp\|hpp\)')
do
  header="$(head -n ${header_length} ${file})"
  # header=$(echo "$header" | sed 's/[[:digit:]]\{4\}-//')
  echo "${header}" > ${temp_header_file}
  diff=$(diff -q "${temp_header_file}" <(echo "$license"))
  if [[ "$diff" != "" ]]
  then
    echo "File \"${file}\" does not have a correct license or year"
    diff "${temp_header_file}" <(echo "$license")

    n_wrong_licenses=$((n_wrong_licenses + 1))
  fi
done
rm -f ${temp_header_file}

exit $n_wrong_licenses
