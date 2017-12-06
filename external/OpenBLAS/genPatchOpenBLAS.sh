#!/bin/sh

# Run this under Elemental's source directory with,
# for example, fileToPatch listed in 'ChangedFiles.txt' and
# fileToPatch.new placed under the directory where this script exists

thisCommand=`basename $0`
fileList=`echo $0 | sed 's/'${thisCommand}'/ChangedFiles.OpenBLAS.txt/'`
patchDir=`echo $0 | sed 's/\/'${thisCommand}'//'`
patchName=OpenBLAS.patch

echo 'fileList = '${fileList}
echo 'patchFile = '${patchDir}/${patchName}

while read f
do
  fdir=`dirname $f`
  forg=`basename $f`
  fnew=${forg}'.new'

  if [ ! -f $f ] || [ ! -f ${patchDir}/${fnew} ] ; then
    continue;
  fi
  pushd . > /dev/null
  cd ${fdir}
  ln -sf ${patchDir}/${fnew} ${fnew}
  popd > /dev/null
  diff -u $f ${fdir}/${fnew}
  rm ${fdir}/${fnew}
done < ${fileList} > ${patchDir}/${patchName}
