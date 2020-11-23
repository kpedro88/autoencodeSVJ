source /cvmfs/sft.cern.ch/lcg/views/LCG_93/x86_64-centos7-gcc7-opt/setup.sh
source $DELPHES_DIR/delphes-env.sh
python -c "import energyflow"
if [ $? -eq 1 ]
then 
    pip install energyflow --user
fi
