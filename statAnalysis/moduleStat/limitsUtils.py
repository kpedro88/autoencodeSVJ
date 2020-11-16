#! /usr/bin/env python
import sys
import os
import commands
import string
import optparse
from Stat.Limits.settings import *



#points = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, vec9, vec10, vec11, vec12, vec13, vec14, vec15, vec16, vec17, vec18, vec19, vec20, vec21,vec22, vec23, vec24, vec25]


class limit(object):
    pass

def readFile(filename, cat):
   
   ifile = open(filename)
   print "Reading ", filename
   v = 0.; o = 0.; u1 = 0.; u2 = 0.; d1 = 0.; d2 = 0.;
   for l in ifile.readlines():
      if l.strip()== "": continue
      else:
         l_split = l.split()
         if l.startswith("Expected 50.0%" ):  v = l_split[4]
         elif l.startswith("Observed" ): o = l_split[4]
         elif l.startswith("Expected  2.5%" ): d2 = l_split[4]
         elif l.startswith("Expected 16.0%" ): d1 = l_split[4]
         elif l.startswith("Expected 84.0%" ): u1 = l_split[4]
         elif l.startswith("Expected 97.5%" ): u2 = l_split[4]


   limits = limit()
   limits.v = v
   limits.o = o
   limits.u1 = u1
   limits.u2 = u2
   limits.d1 = d1
   limits.d2 = d2
   
   return limits

def writeFile(limits, post, outname):
   
   obsline = ("y_observed%s  ") % ( post) 
   medline = ("y_vals%s  ") % ( post) 
   d2line = ("y_down_points2%s  ") % ( post) 
   d1line = ("y_down_points1%s  ") % ( post) 
   u2line = ("y_up_points2%s  ") % ( post) 
   u1line = ("y_up_points1%s  ") % ( post)    


   limitsOrd = collections.OrderedDict(sorted(limits.items()))

   #print limitsOrd
   
   for  k,l in limitsOrd.iteritems(): 

       obsline += ("  %s ") % (l.o)
       medline += ("  %s ") % (l.v)
       d2line += ("  %s ") % (l.d2)
       d1line += ("  %s ") % (l.d1)
       u2line += ("  %s ") % (l.u2)
       u1line += ("  %s ") % (l.u1)

   print obsline
   obsline += ("\n%s\n%s\n%s\n%s\n%s") % (medline, d2line, d1line, u2line, u1line)
   limitfile = open(outname, 'w')
   limitfile.write(obsline)
   limitfile.close()


def getLimits(optdir, post):
      
   i = 0;
   limits = {}
   for item in sigpoints:
         
      mZprime=item[0]
      mDark=item[1]
      rinv=item[2]
      alpha=item[3]

      filename = "%s/SVJ_mZprime%s_mDark%s_rinv%s_alpha%s/asymptotic_mZprime%s_mDark%s_rinv%s_alpha%s%s.log" % \
(optdir, mZprime, mDark, rinv, alpha, mZprime, mDark, rinv, alpha, post)

      limits[float(mZprime)] = readFile(filename, post)

   if not os.path.isdir("data/"): os.system('mkdir data/') 
   writeFile(limits, post, "data/limit"+post+".txt")

 
