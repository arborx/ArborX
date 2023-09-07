#!/usr/bin/env bash

header_length=$(wc -l < LICENSE_FILE_HEADER)
cur_year=$(date +%Y)

license_file="LICENSE_FILE_HEADER"
temp_header_file="scripts/.temp_header"
master="origin/master"

n_wrong_licenses=0
for file in $(git diff --diff-filter=AMRU --name-only ${master}... | grep -e '.*\.\(cc\|cpp\|hpp\)')
do
  header="$(head -n ${header_length} ${file})"
  header=$(echo "$header" | sed 's/[[:digit:]]\{4\}-//')
  echo "${header}" > ${temp_header_file}
  diff=$(diff -q -w "${temp_header_file}" "${license_file}")
  if [[ "$diff" != "" ]]
  then
    echo "File \"${file}\" does not have a correct license or year"
    diff -w "${temp_header_file}" "${license_file}"

    n_wrong_licenses=$((n_wrong_licenses + 1))
  fi
done
rm -f ${temp_header_file}

exit $n_wrong_licenses
