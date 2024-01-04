import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


iris = load_iris()
X, y = iris.data, iris.target


scaler = StandardScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=4))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1, activation='linear'))  # Bu regresyon için çıkış katmanı


model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])


model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))


y_pred = model.predict(X_test)


mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print(f'Mean Absolute Percentage Error (MAPE): {mape:.2f}%')
