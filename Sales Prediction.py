import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

data = {
    'Advertising_Expenditure': [50, 80, 120, 150, 200],
    'Sales': [100, 150, 200, 250, 300]
}

df = pd.DataFrame(data)

sns.scatterplot(x='Advertising_Expenditure', y='Sales', data=df)
plt.title('Advertising Expenditure vs. Sales')
plt.xlabel('Advertising Expenditure')
plt.ylabel('Sales')
plt.show()

X = df[['Advertising_Expenditure']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)


print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.title('Linear Regression Model')
plt.xlabel('Advertising Expenditure')
plt.ylabel('Sales')
plt.show()
