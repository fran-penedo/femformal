from numpy import array


eps = array([  6.76006285e-05,   1.69109226e-04,   1.69109226e-04,
         3.37006811e-04,   3.37006811e-04,   5.02538487e-04,
         5.02538487e-04,   6.66571814e-04,   6.66571814e-04,
         8.26643804e-04,   8.26643804e-04,   9.82973211e-04,
         9.82973211e-04,   1.13717555e-03,   1.13717555e-03,
         1.28730130e-03,   1.28730130e-03,   1.43501387e-03,
         1.43501387e-03,   1.58139466e-03,   1.58139466e-03,
         1.72102231e-03,   1.72102231e-03,   1.86464220e-03,
         1.86464220e-03,   1.91432916e-03,   1.91432916e-03,
         1.99101365e-03,   1.99101365e-03,   1.63818328e-03,
         2.06721041e-03,   1.70543747e-03,   2.15795216e-03,
         1.78098486e-03,   2.24149436e-03,   1.85800162e-03,
         2.35193368e-03,   1.95437646e-03,   2.44826030e-03,
         2.03689297e-03,   2.57332016e-03,   2.15590819e-03,
         2.68955383e-03,   2.25894290e-03,   2.81196799e-03,
         2.37089800e-03,   2.94241339e-03,   2.56910731e-03,
         3.05855973e-03,   3.34166906e-03,   3.64554863e-03,
         3.89798129e-03,   4.66398956e-03,   5.24615377e-03,
         5.76754585e-03,   6.12805255e-03,   6.99884155e-03,
         7.35495187e-03,   8.53474330e-03,   9.96781626e-03,
         1.09717842e-02,   1.25234402e-02,   1.60151652e-02,
         1.76235202e-02,   2.29851978e-02,   2.82642033e-02,
         3.45237329e-02,   3.66049216e-02,   4.98451693e-02,
         7.00292793e-02,   8.37702759e-02,   8.83775292e-02,
         9.33320883e-02,   1.59679517e-01,   2.17497656e-01,
         2.91033589e-01,   3.32575966e-01,   2.99855165e-01,
         2.96063544e-01,   1.08213585e+00])
eta = array([ 0.51244439,  0.51292648,  0.51389046,  0.51533598,  0.51726246,
        0.51966915,  0.52255513,  0.52591925,  0.52976022,  0.53407651,
        0.53886646,  0.54412816,  0.54985955,  0.55605836,  0.56272214,
        0.56984824,  0.57743382,  0.58547585,  0.59397109,  0.60291612,
        0.61230733,  0.6221409 ,  0.63241282,  0.64311887,  0.34869048,
        0.35424988,  0.35991913,  0.36569709,  0.37158257,  0.37757438,
        0.38367128,  0.38987203,  0.39617533,  0.40257989,  0.40908438,
        0.41568744,  0.4223877 ,  0.42918373,  0.43607412,  0.44305742,
        0.45013213,  0.45729675,  0.46454976,  0.47188959,  0.47931468,
        0.48682341,  0.49441416,  0.50208528,  0.95659511,  0.97272941,
        0.9891512 ,  1.00585304,  1.02282728,  1.04006613,  1.0575616 ,
        1.07530556,  1.09328968,  1.11150552,  1.12994444,  1.14859766,
        1.16745627,  1.1865112 ,  1.20575326,  1.2251731 ,  1.24476127,
        1.26450819,  1.28440417,  1.30443942,  1.32460402,  1.34488803,
        1.36528132,  1.38577355,  1.40635469,  1.427016  ,  1.44774317,
        1.46851662,  1.48940741,  1.51013048,  1.73769507,  2.53930302])
nu = array([  0.00000000e+00,   7.46334253e-03,   1.49266503e-02,
         2.23898748e-02,   2.98529405e-02,   3.73157313e-02,
         4.47797040e-02,   5.22445101e-02,   5.97096199e-02,
         6.71747948e-02,   7.46396855e-02,   8.21083846e-02,
         8.95804570e-02,   9.70526237e-02,   1.04528289e-01,
         1.12012819e-01,   1.19497356e-01,   1.26995155e-01,
         1.34497850e-01,   1.42014088e-01,   1.49540556e-01,
         1.57087364e-01,   1.64643678e-01,   1.72235490e-01,
         1.79847262e-01,   1.83918614e-01,   1.87998450e-01,
         1.92097146e-01,   1.96199328e-01,   2.00331005e-01,
         2.04468675e-01,   2.08638964e-01,   2.12826705e-01,
         2.17041126e-01,   2.21294921e-01,   2.25576408e-01,
         2.29895923e-01,   2.34273167e-01,   2.38698348e-01,
         2.43180061e-01,   2.47727668e-01,   2.52351336e-01,
         2.57062060e-01,   2.61871699e-01,   2.66793006e-01,
         2.71839658e-01,   2.77026293e-01,   2.82377261e-01,
         2.87930522e-01,   2.98913035e-01,   3.10834168e-01,
         3.23862684e-01,   3.38158170e-01,   3.53872736e-01,
         3.71085865e-01,   3.89899696e-01,   4.10363498e-01,
         4.32525788e-01,   4.56339618e-01,   4.82094781e-01,
         5.09916702e-01,   5.40012589e-01,   5.72657438e-01,
         6.08229408e-01,   6.47965341e-01,   6.92237062e-01,
         7.42854462e-01,   8.00448544e-01,   8.71149319e-01,
         9.48558717e-01,   1.05352067e+00,   1.23053430e+00,
         1.40102622e+00,   1.54638513e+00,   1.88693144e+00,
         2.30488646e+00,   2.70690733e+00,   2.98935112e+00,
         3.76690891e+00,   5.50460398e+00,   8.04390700e+00])