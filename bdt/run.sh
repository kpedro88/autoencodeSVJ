#!/bin/bash

#  run.sh
#  xAutoencodeSVJ
#
#  Created by Jeremi Niedziela on 16/07/2020.
#  Copyright © 2020 Jeremi Niedziela. All rights reserved.

echo "Sourcing bash profile"
. /Users/Jeremi/.bash_profile

echo "Activating conda"
conda activate ml

cd /Users/Jeremi/Documents/Physics/ETH/autoencodeSVJ/bdt/ || exit

TERM=xterm-color

echo "Running script ${1}"
python $1
