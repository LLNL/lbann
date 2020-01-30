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
