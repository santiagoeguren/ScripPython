################################################################################
#Cargar librerias
################################################################################

import numpy as np

from scipy.stats import norm 

import matplotlib.pyplot as plt




################################################################################
#Calcular
################################################################################




#------------------------------------------------------------------------------
#Probabilidad

 # Calcula P(X>100)
 #Media=100
 #sdt=10
 
1 - norm.cdf(100, loc=100, scale=10) 
norm.cdf(100, loc=100, scale=10) 

#-------------------------------------------------------------------------------
#Cuantieles

norm.ppf(0.025, 0, 1)



#""""
#Fuente:
#https://qu4nt.com/distribucion-normal-manejada-con-python/
#""""


#-------------------------------------------------------------------------------
#Graficar


# Devuelve 100 valores sobre el intervalo [0.001, 0.999]
x = np.linspace(norm.ppf(0.001), norm.ppf(0.999), 100)

# Calcula los valores de la función de densidad de probailidad para x
y = norm.pdf(x)

plt.plot(x, y)  # Crea el objeto gráfico
plt.title('Distribución Normal (0,1)')  # Título del gráfico
plt.ylabel('f(x)')  # Título del eje y
plt.xlabel('X')  # Título del eje x
plt.show()  # Muestra el gráfico

