
###############################################################################
#Cargar librerias
###############################################################################

# importando numpy
import numpy as np 
# importando pandas
import pandas as pd 
# importando scipy.stats
from scipy import stats 
# importando matplotlib
import matplotlib.pyplot as plt 
# importando seaborn
import seaborn as sns 


###############################################################################
#Cargar funciones
###############################################################################


from statsmodels.tsa.arima_model import ARIMA


################################################################################
#Generar Datos
################################################################################

#Definir parámetros
media=0
sdv=0.05
tita_1=0.5
phi_1=0.5
size = 200
x = []
z_t=[]

z_t.append(np.random.normal(media, sdv, 1))

x.append(z_t[0])

#Crear serie de tiempo

#"""
#Para ejecutar el ciclo pinta en azul una linea por debajo
#""""


for t in range(1, size):
    
    z_t.append(np.random.normal(media, sdv, 1)) 

    x.append(tita_1*z_t[t-1]+phi_1*x[t-1]+z_t[t])



x[1:5]


len(x)


t=range(0, size)
len(t)

#-------------------------------------------------------------------------------
#Graficar
#-------------------------------------------------------------------------------

plt.plot(t,x)
plt.show()


plt.scatter(t,x)
plt.show()

################################################################################
#Crear Modelo
################################################################################

arima_model=ARIMA(x,order=(1,0,1))
model=arima_model.fit()
