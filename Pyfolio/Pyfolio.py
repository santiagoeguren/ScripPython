########################################################################
#Cargar librerias
#########################################################################
#Numpy
import numpy as np
#Pandas
import pandas as pd
#Matematicas
import math as math
#Pyfolio
import pyfolio as pf
#Descargar los precios
import yfinance as yf
#Graficar
import matplotlib.pyplot as plt
#pandas_datareader
import pandas_datareader as web
#dates
import matplotlib.dates as dates

###############################################################################
#Descargar datos
################################################################################

#------------------------------------------------------------------------------
#Definir tickers
tickers_list = ['BEN', 'KO', 'NCLH', 'ALK', 'RL']


#-------------------------------------------------------------------------------
#Definir fechas y tipo de precio
stocks = yf.download(tickers_list,'2014-1-1','2019-1-1')['Adj Close']

#------------------------------------------------------------------------------
#Graficar
#------------------------------------------------------------------------------

# Plot all the close prices
((stocks.pct_change()+1).cumprod()).plot(figsize=(10, 7))
# Show the legend
plt.legend()

# Define the label for the title of the figure
plt.title("Returns", fontsize=16)

# Define the labels for x-axis and y-axis
plt.ylabel('Cumulative Returns', fontsize=14)
plt.xlabel('Year', fontsize=14)

# Plot the grid lines
plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
plt.show()


#-------------------------------------------------------------------------------
#Calcular Retornos
#-------------------------------------------------------------------------------

#Estimar los rendimientos diarios
log_ret=np.log(stocks/stocks.shift(1))



################################################################################
#Armar Cartera
################################################################################


#-------------------------------------------------------------------------------
#Cargar pesos

pesos_m=[1/5,1/5,1/5,1/5,1/5]


#-------------------------------------------------------------------------------
#Estimar rendimiento cartera
rendimiento_cartera = pd.Series(np.dot(pesos_m, log_ret.T), index=log_ret.index)

#-------------------------------------------------------------------------------
#Imprimir sumario
pf.create_simple_tear_sheet(rendimiento_cartera)






