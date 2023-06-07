#!/bin/sh

compare_versions()
{
    local v1=( $(echo "$1" | tr '.' ' ') )
    local v2=( $(echo "$2" | tr '.' ' ') )
    local len=$(( ${#v1[*]} > ${#v2[*]} ? ${#v1[*]} : ${#v2[*]} ))
    for ((i=0; i<len; i++))
    do
        [ "${v1[i]:-0}" -gt "${v2[i]:-0}" ] && return 1
        [ "${v1[i]:-0}" -lt "${v2[i]:-0}" ] && return 2
    done
    return 0
}

osx_realpath() {
    [[ $1 = /* ]] && echo "$1" || echo "$PWD/${1#./}"
}

function normpath() {
  # Remove all /./ sequences.
  local path=${1//\/.\//\/}

  # Remove dir/.. sequences.
  while [[ $path =~ ([^/][^/]*/\.\./) ]]; do
    path=${path/${BASH_REMATCH[0]}/}
  done
  echo $path
}

function find_cmake_config_file() {
    local label="$1"
    local center_compiler="$2"
    local lbann_home="$3"

    HOST=$(hostname)
    HOST=${HOST//[[:digit:]]/}
    [[ -z "${SYS_TYPE}" ]] && SYS=${SPACK_ARCH} || SYS="${SYS_TYPE}"

    if [[ "${center_compiler}" =~ .*"%".*"@".* ]]; then
        # Provided compiler has a specific version
        specific_compiler=${center_compiler//%/}
        MATCHED_CONFIG_FILE="LBANN_${HOST}_${label}-${SYS}-${specific_compiler}.cmake"
        MATCHED_CONFIG_FILE_PATH="${lbann_home}/${MATCHED_CONFIG_FILE_PATH}"
    else
        # Only generic family of compiler provided
        generic_compiler=${center_compiler//%/}
        # https://unix.stackexchange.com/questions/240418/find-latest-files
        MATCHED_CONFIG_FILE_PATH=$(find ${lbann_home} -maxdepth 1 -type f -name "LBANN_${HOST}_${label}-${SYS}-${generic_compiler}@*.cmake" -exec stat -c '%X %n' {} \; -print | sort -nr | awk 'NR==1,NR==1 {print $2}')
        if [[ -n "${MATCHED_CONFIG_FILE_PATH}" ]]; then
            MATCHED_CONFIG_FILE=$(basename ${MATCHED_CONFIG_FILE_PATH})
        fi
    fi
    if [[ ! -z "${MATCHED_CONFIG_FILE}" ]]; then
        if [[ ! -e "${MATCHED_CONFIG_FILE_PATH}" || ! -r "${MATCHED_CONFIG_FILE_PATH}" ]]; then
            echo "INFO: Unable to open the generated config file: ${MATCHED_CONFIG_FILE} at ${lbann_home}"
        fi
    else
        echo "INFO: Unable to find a generated config file for: ${LBANN_LABEL} ${CENTER_COMPILER} in ${lbann_home}"
    fi
}
