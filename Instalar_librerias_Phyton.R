########################################################################
#Cargar librerias
#########################################################################
#Conecta R y Phton
library(reticulate)




#installar por primera 
install_miniconda(path = miniconda_path(), update = TRUE, force = FALSE)

#reinstalar
reticulate::install_miniconda(force = TRUE)








#Pandas y scikit-learn
py_install(packages = c("pandas", "scikit-learn"))

#"scikit-plot
py_install(packages = c("scikit-plot"))

#py_install(packages = c("matplotlib.pyplot"))

py_install(packages = c("seaborn"))

py_install(packages = "statsmodels")

#Pyfolio
py_install(packages = c("pyfolio"))

#mkl-service
py_install(packages = c("mkl-service"))

#yfinance
py_install(packages = c("yfinance"))


#pycaret[full]
py_install(packages = c("pycaret[full]"))









