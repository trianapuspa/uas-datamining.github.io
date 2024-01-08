import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle

df = pd.read_csv('dataset/calonpembelimobil.csv')

df = df.drop('ID', axis=1)

scaler = StandardScaler()
df[['Usia', 'Status', 'Kelamin', 'Memiliki_Mobil', 'Penghasilan']] = scaler.fit_transform(df[['Usia', 'Status', 'Kelamin', 'Memiliki_Mobil', 'Penghasilan']])

feature_columns = ['Usia', 'Status', 'Kelamin', 'Memiliki_Mobil', 'Penghasilan']
X = df[feature_columns]
y = df['Beli_Mobil']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

with open('decision_tree_model.pkl', 'wb') as model_file:
    pickle.dump(dt_model, model_file)

lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

with open('logistic_regression_model.pkl', 'wb') as model_file:
    pickle.dump(lr_model, model_file)
