#!/bin/bash

re=.*\.py$

main () {

  #echo "main!!"
  #echo $1

  for f in `ls $1`; do
    #echo $f
    if [ -d $1/${f} ]; then
      #echo d ${f}
      main $1/$f
    elif [ -f $1/${f} ]; then
      if [[ ${f} =~ ${re} ]]; then
        #echo f ${f}
        expand -t 8 $1/${f} > /tmp/e && cp /tmp/e $1/${f}
        echo -e "\t$1/${f}"
      fi
    else
      echo else $f
    fi
  done

}

echo "Replacing tabs by spaces in the following files:"
main ../autoencodeSVJ
