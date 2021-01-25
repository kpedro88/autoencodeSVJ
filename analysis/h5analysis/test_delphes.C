HDF5 "test_delphes.h5" {
GROUP "/" {
   GROUP "event_features" {
      DATASET "data" {
         DATATYPE  H5T_IEEE_F64LE
         DATASPACE  SIMPLE { ( 6, 5 ) / ( 6, 5 ) }
         DATA {
         (0,0): 124.028, -3.36592, 0.0277447, 4386.96, 4274.87,
         (1,0): 25.843, -3.66084, 0.496769, 2487.17, 2460.89,
         (2,0): 220.299, 3.74044, 2.20449, 1714.05, 1297.87,
         (3,0): 21.7602, 2.1516, -0.452918, 2061.68, 2040.32,
         (4,0): 16.8975, 2.99752, -1.43669, 1768.42, 1751.97,
         (5,0): 138.131, 4.34066, 2.60967, 2013.27, 1868.5
         }
      }
      DATASET "labels" {
         DATATYPE  H5T_STRING {
            STRSIZE H5T_VARIABLE;
            STRPAD H5T_STR_NULLTERM;
            CSET H5T_CSET_UTF8;
            CTYPE H5T_C_S1;
         }
         DATASPACE  SIMPLE { ( 5 ) / ( 5 ) }
         DATA {
         (0): "MET", "METEta", "METPhi", "MT", "Mjj"
         }
      }
   }
   GROUP "jet_constituents" {
      DATASET "data" {
         DATATYPE  H5T_IEEE_F64LE
         DATASPACE  SIMPLE { ( 6, 2, 100, 5 ) / ( 6, 2, 100, 5 ) }
         DATA {
         (0,0,0,0): -1.65668, 0.388567, 243.848, -1.65668, 662.374,
         (0,0,1,0): -1.65639, 0.385041, 188.485, -1.65639, 511.846,
         ...
         (0,0,33,0): -1.76551, 0.440728, 0.443774, -1.76551, 1.33479,
         (0,0,34,0): 0, 0, 0, 0, 0,
         ...
         (0,0,99,0): 0, 0, 0, 0, 0,
         (0,1,0,0): 1.82933, -2.88267, 144.525, 1.82933, 461.775,
         ...
         }
      }
      DATASET "labels" {
         DATATYPE  H5T_STRING {
            STRSIZE H5T_VARIABLE;
            STRPAD H5T_STR_NULLTERM;
            CSET H5T_CSET_UTF8;
            CTYPE H5T_C_S1;
         }
         DATASPACE  SIMPLE { ( 5 ) / ( 5 ) }
         DATA {
         (0): "Eta", "Phi", "PT", "Rapidity", "Energy"
         }
      }
   }
   GROUP "jet_eflow_variables" {
      DATASET "data" {
         DATATYPE  H5T_IEEE_F64LE
         DATASPACE  SIMPLE { ( 6, 2, 13 ) / ( 6, 2, 13 ) }
         DATA {
         (0,0,0): 1, 0.0198774, 0.00298838, 0.00103478, 0.00166784,
         (0,0,5): 0.000543013, 7.05435e-05, 0.000519467, 5.41062e-05,
         (0,0,9): 0.00039511, 5.94012e-05, 3.31523e-05, 7.85374e-06,
         (0,1,0): 1, 0.170068, 0.0659148, 0.027996, 0.0393802, 0.0158332,
         (0,1,6): 0.00472378, 0.0116867, 0.0073004, 0.028923, 0.01121,
         (0,1,11): 0.0066973, 0.00491886,
         ...
         (5,1,0): 1, 0.16693, 0.04395, 0.0144555, 0.032751, 0.00967652,
         (5,1,6): 0.00620784, 0.00776681, 0.00611702, 0.0278655, 0.00733655,
         (5,1,11): 0.00546712, 0.00465158
         }
      }
      DATASET "labels" {
         DATATYPE  H5T_STRING {
            STRSIZE H5T_VARIABLE;
            STRPAD H5T_STR_NULLTERM;
            CSET H5T_CSET_UTF8;
            CTYPE H5T_C_S1;
         }
         DATASPACE  SIMPLE { ( 13 ) / ( 13 ) }
         DATA {
         (0): "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11",
         (12): "12"
         }
      }
   }
   GROUP "jet_features" {
      DATASET "data" {
         DATATYPE  H5T_IEEE_F64LE
         DATASPACE  SIMPLE { ( 6, 2, 9 ) / ( 6, 2, 9 ) }
         DATA {
         (0,0,0): -1.6552, 0.387106, 988.844, 14.5396, 0.56, 0.386797,
         (0,0,6): 0.00343321, 2, 2682.37,
         (0,1,0): 1.80032, -2.94138, 548.159, 93.9983, 0.764706, 0.29298,
         (0,1,6): 0.0146512, 1, 1706.5,
         ...
         }
      }
      DATASET "labels" {
         DATATYPE  H5T_STRING {
            STRSIZE H5T_VARIABLE;
            STRPAD H5T_STR_NULLTERM;
            CSET H5T_CSET_UTF8;
            CTYPE H5T_C_S1;
         }
         DATASPACE  SIMPLE { ( 9 ) / ( 9 ) }
         DATA {
         (0): "Eta", "Phi", "Pt", "M", "ChargedFraction", "PTD", "Axis2",
         (7): "Flavor", "Energy"
         }
      }
   }
}
}
