import matplotlib
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResultsWrapper
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from utils import *
import warnings
warnings.filterwarnings("ignore")
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

#ucitavanje podataka:
podaci = pd.read_csv('C:\\Users\\Korisnik\\Desktop\\anja\\nans_novi\\NANS_2\\data\\dog_breeds.csv', sep = ',')
#podaci_test = pd.read_csv('C:\\Users\\Korisnik\\Desktop\\anja\\nans_novi\\NANS_2\\data\\dog_breeds_test.csv', sep = ',')
#podaci_train = pd.read_csv('C:\\Users\\Korisnik\\Desktop\\anja\\nans_novi\\NANS_2\\data\\dog_breeds_train.csv', sep = ',')
#print(podaci1.head())

#print(podaci.isnull().sum()) #nema null podataka
#pretprocesiranje
podaci.fillna(podaci, inplace=True)

matplotlib.rcParams['figure.figsize'] = (8, 4)
sb.set(font_scale=1.)

#predvidjanje max_life_expectancy
y1 = podaci['max_life_expectancy']
x1 = podaci.drop(columns=['max_life_expectancy','Name'])

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.2, shuffle=True,random_state=42)

#Kreiranje i treniranje modela
# model = LinearRegression()
# model.fit(x1_train, y1_train)
# model = get_fitted_model(x1_train,y1_train)
# print(model.summary())


x_with_const = sm.add_constant(x1)
model = sm.OLS(y1, x_with_const).fit()
print(model.summary())
#Predviđanje
y1_pred = model.predict(x_with_const)

# Evaluacija modela
rmse = get_rmse(y1, y1_pred)
print("rmse:", rmse)
print(are_assumptions_satisfied(model,x1_train,y1_train))
if (are_assumptions_satisfied(model,x1_train,y1_train) == True):
    print("Sve L.I.N.E pretpostavke su zadovoljene.")

# Vizualizacija predviđenih vrednosti u poređenju sa stvarnim vrednostima
#residuals = y1_test - y1_pred
residuals = y1 - y1_pred
plt.scatter(y1_pred, residuals)
plt.xlabel('Predviđene vrednosti')
plt.ylabel('Reziduali')
plt.axhline(y=0, color='r', linestyle='-')
plt.title('Grafikon reziduala')
plt.show()


#predvidjanje min_life_expectancy u odnosu na
#max_height_male,max_height_female,max_weight_male,max_weight_female,min_height_male,min_height_female,min_weight_male,min_weight_female
#playfulness,protectiveness,trainability,energy
y2 = podaci['min_life_expectancy']
x2 = podaci[['max_height_male','max_height_female','max_weight_male','max_weight_female','min_height_male','min_height_female','min_weight_male','min_weight_female','playfulness','protectiveness','trainability','energy']]

# Podela podataka na trening i test skupove
x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.2, random_state=42)

# Normalizacija ili standardizacija podataka
scaler = StandardScaler()
x2_train_scaled = scaler.fit_transform(x2_train)
x2_test_scaled = scaler.transform(x2_test)

# Kreiranje i treniranje Lasso modela
lasso_model1 = Lasso(alpha=0.1)  # alfa za sad 0.1
lasso_model1.fit(x2_train_scaled, y2_train)
lasso_model2 = Lasso(alpha=0.2)  # alfa 0.2
lasso_model2.fit(x2_train_scaled, y2_train)
lasso_model3 = Lasso(alpha=0.5)  # alfa 0.5
lasso_model3.fit(x2_train_scaled, y2_train)

# Predviđanje na test skupu
y2_pred1 = lasso_model1.predict(x2_test_scaled)
y2_pred2 = lasso_model2.predict(x2_test_scaled)
y2_pred3 = lasso_model3.predict(x2_test_scaled)

# Evaluacija modela
rmse_lasso1 = get_rmse(y2_test, y2_pred1)
print("RMSE Lasso regresije alpha = 0.1:", rmse_lasso1)
rmse_lasso2 = get_rmse(y2_test, y2_pred2)
print("RMSE Lasso regresije alpha = 0.2:", rmse_lasso2)
rmse_lasso3 = get_rmse(y2_test, y2_pred3)
print("RMSE Lasso regresije alpha = 0.5:", rmse_lasso3)

# Vizualizacija predviđenih vrednosti u poređenju sa stvarnim vrednostima
plt.scatter(y2_test, y2_pred1, c='red')
plt.scatter(y2_test, y2_pred2, c='blue')
plt.scatter(y2_test, y2_pred3, c='yellow')
plt.xlabel('Stvarni minimalni životni vek')
plt.ylabel('Predviđeni minimalni životni vek')
plt.title('Predikcija minimalnog životnog veka pasa (Lasso regresija)')
plt.show()

#prosek i broj pasa
df = podaci.copy()
df['avg_life_expectancy'] = (df['min_life_expectancy'] + df['max_life_expectancy']) / 2

df.drop(columns=['min_life_expectancy', 'max_life_expectancy'], inplace=True)

#plt.hist(podaci['max_life_expectancy'], bins=16, edgecolor='black')
plt.hist(df['avg_life_expectancy'], bins=16, edgecolor='black')
plt.title('Histogram prosecnog životnog veka pasa')
plt.xlabel('Životni vek')
plt.ylabel('Broj pasa')
plt.show()

#predvidjanje
y4 = podaci['max_life_expectancy']
x4 = podaci[['max_height_female','max_weight_female','min_height_female','min_weight_female']]

x4_train, x4_test, y4_train, y4_test = train_test_split(x4, y4, test_size=0.2, shuffle=True,random_state=42)

# x_with_const4 = sm.add_constant(x4)
# model4 = sm.OLS(y4, x_with_const4).fit()
# print(model4.summary())
# # Predviđanje
# y4_pred = model4.predict(x_with_const4)
model4 = LinearRegression()
model4.fit(x4_train, y4_train)

# Predviđanje na test skupu
y4_pred = model4.predict(x4_test)
# Evaluacija modela
#adj_r2 = get_rsquared_adj(model4,x4_test,y4_test)
#print("r2:", adj_r2)
mse = mean_squared_error(y4_test, y4_pred)
print("Mean Squared Error:", mse)
# print(are_assumptions_satisfied(model4,x4_train,y4_train))
# if (are_assumptions_satisfied(model4,x4_train,y4_train) == True):
#     print("Sve L.I.N.E pretpostavke su zadovoljene.")

# Vizualizacija
residuals4 = y4_test - y4_pred
plt.scatter(y4_pred, residuals4)
plt.xlabel('Predviđene vrednosti')
plt.ylabel('Reziduali')
plt.axhline(y=0, color='r', linestyle='-')
plt.title('Grafikon reziduala')
plt.show()

#predvidjanje 5
y5 = podaci['min_life_expectancy']
x5 = podaci[['max_height_male','max_weight_male','min_height_male','min_weight_male']]

x5_train, x5_test, y5_train, y5_test = train_test_split(x5, y5, test_size=0.2, shuffle=True,random_state=42)

model5 = LinearRegression()
model5.fit(x5_train, y5_train)

# Predviđanje na test skupu
y5_pred = model5.predict(x5_test)
# Evaluacija modela
mse = get_mse(y5_test, y5_pred)
print("Mean Squared Error:", mse)


# Vizualizacija
residuals5 = y5_test - y5_pred
plt.scatter(y5_pred, residuals5, c = 'blue')
plt.xlabel('Predviđene vrednosti')
plt.ylabel('Reziduali')
plt.axhline(y=0, color='red', linestyle='-')
plt.title('Grafikon reziduala')
plt.show()

