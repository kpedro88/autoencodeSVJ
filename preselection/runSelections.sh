#!/bin/bash

nEvents=$1
i=$(($2-1))

cd /afs/cern.ch/work/j/jniedzie/private/svjets/autoencodeSVJ/preselection
. setenv.sh

ir=$(($i / 6))
im=$(($i % 6))

masses=(1500 2000 2500 3000 3500 4000)
rinvs=(15 30 45 60 75)

mass=${masses[$im]}
rinv=${rinvs[$ir]}

echo "Running for mass: ${mass}, r_inv: ${rinv}"

./SVJselection inputFileLists/input_file_list_m${mass}_r${rinv}.txt SVJ_m${mass}_r${rinv} results/ 0 $nEvents
