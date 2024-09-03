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

function host_basename() {
    HOST=$(hostname)
    HOST=${HOST//[[:digit:]]/}
    HOST=${HOST//\-/}
    echo ${HOST}
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
    local lbann_build_dir="$3"

    HOST=$(hostname)
    HOST=${HOST//[[:digit:]]/}
    HOST=$(echo $HOST | sed 's/\(.*\)-$/\1/')

    [[ -z "${SYS_TYPE}" ]] && SYS=${SPACK_ARCH} || SYS="${SYS_TYPE}"

    if [[ "${center_compiler}" =~ .*"%".*"@".* ]]; then
        # Provided compiler has a specific version
        specific_compiler=${center_compiler//%/}
        MATCHED_CONFIG_FILE="LBANN_${HOST}_${label}-${SYS}-${specific_compiler}.cmake"
        MATCHED_CONFIG_FILE_PATH="${lbann_build_dir}/${MATCHED_CONFIG_FILE}"
    else
        # Only generic family of compiler provided
        generic_compiler=${center_compiler//%/}
        # https://unix.stackexchange.com/questions/240418/find-latest-files
        # OS X and Linux have different flags for the stat call
        SYS_UNAME=$(uname -s)
        if [[ ${SYS_UNAME} = "Darwin" ]]; then
            MATCHED_CONFIG_FILE_PATH=$(find ${lbann_build_dir} -maxdepth 1 -type f -name "LBANN_${HOST}_${label}-${SYS}-${generic_compiler}@*.cmake" -exec stat -f '%a %N' {} \; -print | sort -nr | awk 'NR==1,NR==1 {print $2}')
        else
            MATCHED_CONFIG_FILE_PATH=$(find ${lbann_build_dir} -maxdepth 1 -type f -name "LBANN_${HOST}_${label}-${SYS}-${generic_compiler}@*.cmake" -exec stat -c '%X %n' {} \; -print | sort -nr | awk 'NR==1,NR==1 {print $2}')
        fi
        if [[ -n "${MATCHED_CONFIG_FILE_PATH}" ]]; then
            MATCHED_CONFIG_FILE=$(basename ${MATCHED_CONFIG_FILE_PATH})
        fi
    fi
    if [[ ! -z "${MATCHED_CONFIG_FILE}" ]]; then
        if [[ ! -e "${MATCHED_CONFIG_FILE_PATH}" || ! -r "${MATCHED_CONFIG_FILE_PATH}" ]]; then
            echo "INFO: Unable to open the generated config file: ${MATCHED_CONFIG_FILE} at ${lbann_build_dir}"
        fi
    else
        echo "INFO: Unable to find a generated config file for: ${LBANN_LABEL} ${CENTER_COMPILER} in ${lbann_build_dir}"
    fi
}

function update_LBANN_DEPENDENT_MODULES_field() {
    local p="$1"

    if [[ (! "${LBANN_DEPENDENT_MODULES:-}" =~ .*"${p}".*) ]]; then
        echo "INFO: Adding package ${p}"
        if [[ -z "${LBANN_DEPENDENT_MODULES:-}" ]]; then
            LBANN_DEPENDENT_MODULES="${p}"
        else
            LBANN_DEPENDENT_MODULES="${p};${LBANN_DEPENDENT_MODULES}"
        fi
    else
        echo "WARNING: Skipping package ${p} which is already in ${LBANN_DEPENDENT_MODULES}"
    fi
}
