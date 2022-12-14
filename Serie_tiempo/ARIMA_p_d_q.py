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
#AutoSarima
import pmdarima as pm


###############################################################################
#Cargar funciones
###############################################################################


from statsmodels.tsa.arima.model  import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


################################################################################
#Fuente
################################################################################
#https://www.justintodata.com/arima-models-in-python-time-series-prediction/
#https://towardsdatascience.com/forecast-with-arima-in-python-more-easily-with-scalecast-35125fc7dc2e

################################################################################
#Generar Datos
################################################################################

%reset
plt.clf()

media=0
sdv=1
beta=0.05



size = 100
x = []
tendencia=[]



z_t.append(np.random.normal(media, sdv, 1))
x.append(z_t[0]+10)
tendencia.append(0)




for t in range(1, size):
    
    z_t.append(np.random.normal(media, sdv, 1)) 

    tendencia.append(t*beta)

    x.append(tendencia[t]+z_t[t]+10)
    
    
 
t=range(0, size)
len(t)

#Tranformar los datos en pandas data frame

y=pd.DataFrame(x) 

y.columns =["serie"]



y.plot()
plt.show()

print(x)

################################################################################
#Crear Modelo
################################################################################







x = np.log(y) # don't forget to transform the data back when making real predictions



x.plot()
plt.show()

#Divir entre train y test


msk = (x.index < len(x)-30)
x_train = x[msk].copy()
x_test = x[~msk].copy()



#Diferenciar

dx_train = x_train.diff().dropna()
dx_train.plot()
plt.show()

help(ARIMA)


acf = plot_acf(dx_train)
plt.show()

pacf = plot_pacf(dx_train)
plt.show()


#-------------------------------------------------------------------------------
#Determinar ARIMA

model = ARIMA(x_train, order=(0,1,0),trend="t")
model_fit = model.fit()
print(model_fit.summary())


#-------------------------------------------------------------------------------
#Residuos

residuals = model_fit.resid[1:]
fig, ax = plt.subplots(1,2)
residuals.plot(title='Residuals', ax=ax[0])
residuals.plot(title='Density', kind='kde', ax=ax[1])
plt.show()


acf_res = plot_acf(residuals)
plt.show()

pacf_res = plot_pacf(residuals)
plt.show()

#-------------------------------------------------------------------------------
#Hacer prediccion

forecast_test = model_fit.forecast(len(x_test))

x['forecast_manual'] = [None]*len(x_train) + list(forecast_test)

x.plot()
plt.show()

#-------------------------------------------------------------------------------
#Forma  automatica

auto_arima = pm.auto_arima(x_train, stepwise=False, seasonal=False)
auto_arima

auto_arima.summary()

forecast_test_auto = auto_arima.predict(n_periods=len(x_test))
x['forecast_auto'] = [None]*len(x_train) + list(forecast_test_auto)

x.plot()
plt.show()

