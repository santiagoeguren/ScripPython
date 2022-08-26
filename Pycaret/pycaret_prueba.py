
#https://towardsdatascience.com/introduction-to-binary-classification-with-pycaret-a37b3e89ad8d

from pycaret.datasets import get_data
dataset = get_data('credit')


# check the shape of data
dataset.shape

# sample 5% of data to be used as unseen data
data = dataset.sample(frac=0.95, random_state=786)
data_unseen = dataset.drop(data.index)
data.reset_index(inplace=True, drop=True)
data_unseen.reset_index(inplace=True, drop=True)

# print the revised shape
print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))


from pycaret.classification import *

exp_clf101 = setup(data = data, target = 'LIMIT_BAL', fold_shuffle=False,session_id=123)


s = setup(data = data, target = 'default')

best_model = compare_models(s)

dt = create_model('dt')








from pycaret.datasets import get_data
dataset = get_data('diamond')







data = dataset.sample(frac=0.002, random_state=786).reset_index(drop=True)
data_unseen = dataset.drop(data.index).reset_index(drop=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))

from pycaret.regression import *


exp_reg101 = setup(data = data, target = 'Price') 

clf1 = setup(data = dataset, target = 'Price',train_size = 0.7, session_id = 2, shuffle=True)

clf1 = setup(data = dataset, target = 'Price', train_size = 0.7, data_split_shuffle=False, session_id = 5)


setup(data = dataset, target = 'Price')

# load dataset
from pycaret.datasets import get_data
data = get_data('iris')


# init setup
from pycaret.classification import *
s = setup(data, target = 'species', session_id = 1,fold_shuffle=True)

# train model
lr = create_model('lr')# generate dashboard
dashboard(lr)

eda()




import numpy as np
from sklearn.model_selection import KFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])
kf = KFold(n_splits=2,shuffle=True)
kf.get_n_splits(X)
