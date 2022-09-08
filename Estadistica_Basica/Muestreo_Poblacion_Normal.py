########################################################################
#Cargar librerias
#########################################################################

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



################################################################################
#Analisis estadistico
################################################################################

#"""
#Fuente
##https://relopezbriega.github.io/blog/2015/06/27/probabilidad-y-estadistica-con-python/
#""""

#-------------------------------------------------------------------------------
#Generar datos
#-------------------------------------------------------------------------------

# para poder replicar el random
np.random.seed(2131982) 


# datos normalmente distribuidos

#"""
#filas y columnas
#""""

datos = np.random.randn(5, 4) 
datos

#-------------------------------------------------------------------------------
# Estimaciones
#-------------------------------------------------------------------------------

# Calcula la media aritmetica de stats
datos.mean() 

#-------------------------------------------------------------------------------
#Calculos basados mayormente en numpy

# Mismo resultado desde la funcion de numpy
np.mean(datos) 


# media aritmetica de cada fila
datos.mean(axis=1) 

# media aritmetica de cada columna
datos.mean(axis=0) 

# mediana
np.median(datos) 

 # media aritmetica de cada columna
np.median(datos, 0)


# Desviación típica
np.std(datos)


# Desviación típica de cada columna
np.std(datos, 0)

# varianza
np.var(datos) 

# varianza de cada columna
np.var(datos, 0)


# moda
# Calcula la moda de cada columna
stats.mode(datos) 

#""""
# el 2do array devuelve la frecuencia.
#""""


datos2 = np.array([1, 2, 3, 6, 6, 1, 2, 4, 2, 2, 6, 6, 8, 10, 6])
stats.mode(datos2)

#"""
# aqui la moda es el 6 porque aparece 5 veces en el vector.
#""""


# correlacion
np.corrcoef(datos) 
#"""
## Crea matriz de correlación.
#"""

# calculando la correlación entre dos vectores.
np.corrcoef(datos[0], datos[1])

# covarianza
np.cov(datos) 
#"""
## calcula matriz de covarianza
#""""

# covarianza de dos vectores
np.cov(datos[0], datos[1])


# Calculando la simetria con scipy
stats.skew(datos)

#-------------------------------------------------------------------------------
#Calculos basados mayormente con pandas

# usando pandas
dataframe = pd.DataFrame(datos, index=['a', 'b', 'c', 'd', 'e'], 
                        columns=['col1', 'col2', 'col3', 'col4'])
dataframe


# resumen estadistadistico con pandas
dataframe.describe()


# sumando las columnas
dataframe.sum()


# sumando filas
dataframe.sum(axis=1)


# acumulados
dataframe.cumsum() 

# media aritmetica de cada columna con pandas
dataframe.mean()

dataframe.mean(axis=1)

################################################################################
#Graficos
################################################################################

#parametros esteticos de seaborn
sns.set_palette("deep", desat=.6)
sns.set_context(rc={"figure.figsize": (8, 4)})

#-------------------------------------------------------------------------------
#Histograma
#-------------------------------------------------------------------------------



# media y desvio estandar
mu, sigma = 0, 0.1 
#creando muestra de datos
s = np.random.normal(mu, sigma, 1000) 


#-------------------------------------------------------------------------------
#1 Forma
#-------------------------------------------------------------------------------

#""""
#No funciona
#""""

%matplotlib inline 

# histograma de distribución normal.
cuenta, cajas, ignorar = plt.hist(s, 30, normed=True)
normal = plt.plot(cajas, 1/(sigma * np.sqrt(2 * np.pi)) *
         np.exp( - (cajas - mu)**2 / (2 * sigma**2) ),
         linewidth=2, color='r')
         
#-------------------------------------------------------------------------------
#2 Forma
#-------------------------------------------------------------------------------    
         
plt.title('Histograma')
plt.hist(s, bins = 60)
plt.grid(True)
plt.show()
plt.clf()       
     
     
#-------------------------------------------------------------------------------
#3 Forma
#-------------------------------------------------------------------------------      
     

sns.distplot(a=s, color='red',
             hist_kws={"edgecolor": 'white'})
 
plt.show()
         
      


#-------------------------------------------------------------------------------
#Box-plot
#------------------------------------------------------------------------------- 
 
# Creating dataset
np.random.seed(10)
 
data_1 = np.random.normal(100, 10, 200)
data_2 = np.random.normal(90, 20, 200)
data_3 = np.random.normal(80, 30, 200)
data_4 = np.random.normal(70, 40, 200)
data = [data_1, data_2, data_3, data_4]
 
fig = plt.figure(figsize =(10, 7))
 
# Creating axes instance
ax = fig.add_axes([0, 0, 1, 1])
 
# Creating plot
bp = ax.boxplot(data)
 
# show plot
plt.show()


#-------------------------------------------------------------------------------
#Histograma sobre la distribucion
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
#Una distrubucion

#Estilo
sns.set(style='ticks')


# parameterise our distributions
d1 = sps.norm(0, 10)

# sample values from above distributions
y1 = d1.rvs(300)


# create new figure with size given explicitly
plt.figure(figsize=(10, 6))


# add histogram showing individual components
plt.hist(y1, 31, histtype='barstacked', density=True, alpha=0.4, edgecolor='none')


# get X limits and fix them
mn, mx = plt.xlim()
plt.xlim(mn, mx)

# add our distributions to figure
x = np.linspace(mn, mx, 301)
plt.plot(x, d1.pdf(x) * (len(y1) / len(y1)), color='C0', ls='--', label='d1')


# finish up
plt.legend()
plt.ylabel('Probability density')
sns.despine()

# show plot
plt.show()



#-------------------------------------------------------------------------------
#dos distrubucion



# parameterise our distributions
d1 = sps.norm(0, 10)
d2 = sps.norm(60, 15)

# sample values from above distributions
y1 = d1.rvs(300)
y2 = d2.rvs(200)
# combine mixture
ys = np.concatenate([y1, y2])

# create new figure with size given explicitly
plt.figure(figsize=(10, 6))

# add histogram showing individual components
plt.hist([y1, y2], 31, histtype='barstacked', density=True, alpha=0.4, edgecolor='none')

# get X limits and fix them
mn, mx = plt.xlim()
plt.xlim(mn, mx)

# add our distributions to figure
x = np.linspace(mn, mx, 301)
plt.plot(x, d1.pdf(x) * (len(y1) / len(ys)), color='C0', ls='--', label='d1')
plt.plot(x, d2.pdf(x) * (len(y2) / len(ys)), color='C1', ls='--', label='d2')

# estimate Kernel Density and plot
kde = sps.gaussian_kde(ys)
plt.plot(x, kde.pdf(x), label='KDE')

# finish up
plt.legend()
plt.ylabel('Probability density')
sns.despine()

# show plot
plt.show()
