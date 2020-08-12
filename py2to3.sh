#!/bin/bash

re=.*\.py$

main () {

  #echo "main!!"
  #echo $1

  for f in `ls $1`; do
    #echo $f
    if [ -d $1/${f} ]; then
      echo ""
      echo "================================================================="
      echo "In dir ${f}"
      main $1/$f
    elif [ -f $1/${f} ]; then
      if [[ ${f} =~ ${re} ]]; then
        echo ""
        echo "Migrate file ${f}"
        #2to3-3 -w -n $1/${f}
        2to3 -w -n $1/${f}
      fi
    else
      echo "else $f"
    fi
  done

}

main ../autoencodeSVJ
