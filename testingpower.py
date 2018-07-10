import pandas
from pandas.plotting import scatter_matrix
import quandl, math, datetime
import time
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle


style.use('ggplot')

#create dataset
filename = 'household_power_consumption.txt'
names = [ 'Global_active_power', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
dataset = pandas.read_csv(filename, sep=';', low_memory=False, names=names)
print(dataset.shape)
#print(dataset.head(20))
dataset['Global_active_power'] = pandas.to_numeric(dataset['Global_active_power'], errors='coerce')
dataset['Sub_metering_1'] = pandas.to_numeric(dataset['Sub_metering_1'], errors='coerce')
dataset['Sub_metering_2'] = pandas.to_numeric(dataset['Sub_metering_2'], errors='coerce')
dataset['Sub_metering_3'] = pandas.to_numeric(dataset['Sub_metering_3'], errors='coerce')
dataset['active_energy_consumed'] = (((dataset['Global_active_power'])*1000.0)/60.0 - dataset['Sub_metering_1'] - dataset['Sub_metering_1'] - dataset['Sub_metering_1'])


#print(dataset.head(20))


forecast_col = 'active_energy_consumed'
dataset.fillna(-99999, inplace=True)
forecast_out = int(math.ceil(0.00005*len(dataset)))
print(forecast_out)
dataset['label'] = dataset[forecast_col].shift(-forecast_out)
#print(dataset.head(20))

X = np.array(dataset.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out:]
dataset.dropna(inplace=True)

y = np.array(dataset['label'])
#print(len(X),len(y))

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)


clf = LinearRegression(n_jobs=1)
clf.fit(X_train, y_train)

with open('linearregression.pickle','wb') as f:
    pickle.dump(clf, f)

pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in)


accuracy = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy ,forecast_out)



dataset['active_energy_consumed'].plot()
plt.show()
dataset['Global_active_power'].plot()
plt.show()
dataset['active_energy_consumed'].plot()
dataset['Global_active_power'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Consumption')
plt.show()























