#!/bin/bash

#  run.sh
#  xAutoencodeSVJ
#
#  Created by Jeremi Niedziela on 16/07/2020.
#  Copyright © 2020 Jeremi Niedziela. All rights reserved.



echo "Sourcing bash profile"
. ~/.bash_profile

echo "Activating conda"
conda activate ml

cd /Users/Jeremi/Documents/Physics/ETH/autoencodeSVJ
cd autoencode/module

echo "Running setup script"
python setup.py install --user;

TERM=xterm-color

cd ../../
echo "Running test script"
python $1
