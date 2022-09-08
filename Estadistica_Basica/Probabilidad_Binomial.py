################################################################################
#Cargar librerias
################################################################################

import numpy as np

from scipy.stats import binom

import matplotlib.pyplot as plt
# importando seaborn
import seaborn as sns 

from numpy import random



#""""
#Fuente:
#https://qu4nt.com/distribucion-normal-manejada-con-python/
#""""

################################################################################
#Calcular
################################################################################




#------------------------------------------------------------------------------
#Probabilidad


#P[X=10]´

binom.pmf (k = 10 , n = 12 , p = 0.6 )

#P[X<=2]

binom.cdf (k = 2 , n = 5 , p = 0.5 )

#-------------------------------------------------------------------------------
#Cuantieles

binom.ppf(0.025,  n = 5 , p = 0.5)



#-------------------------------------------------------------------------------
#Graficar

x = random.binomial (n = 10 , p = 0.5 , size = 1000 )

sns.distplot (x, hist = True , kde = False )

plt.show ()

