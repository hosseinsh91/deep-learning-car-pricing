# Car Price Prediction Using Deep Learning

## **📌 Project Overview**
This project builds a **deep learning regression model** using **TensorFlow/Keras** to predict **car prices** based on various vehicle attributes. The dataset includes details such as **engine type, fuel type, car dimensions, horsepower, and more**. The project preprocesses data, applies feature encoding, and experiments with multiple loss functions for model optimization.

### **🚀 Key Features**
✅ **Data Preprocessing (Handling Missing Values, Encoding, Scaling)**  
✅ **Feature Engineering for Regression Tasks**  
✅ **Neural Network Model with ReLU Activation**  
✅ **Comparison of MSE, MAE, and Custom Loss Function**  
✅ **Training Performance Visualization**  

---

## **📌 Dataset Description**
The dataset contains **205 rows** and **26 features**, including:
- **Categorical Attributes**: Fuel Type, Aspiration, Car Body, Drive Wheel, etc.
- **Numerical Attributes**: Wheelbase, Car Length, Car Width, Horsepower, Price, etc.
https://www.kaggle.com/hellbuoy/car-price-prediction
### **📌 Sample Columns**
| symboling | fueltype | carbody | wheelbase | carwidth | horsepower | price |
|-----------|---------|--------|----------|---------|-----------|------|
| 3         | gas     | sedan  | 88.6     | 64.1    | 111       | 13495 |
| 2         | diesel  | hatchback | 94.5   | 65.2    | 154       | 16500 |

---

## **📌 Data Preprocessing**
### 
**1️⃣ Removing Unnecessary Columns**
```python
df = df.drop(['car_ID', 'CarName'], axis=1)
```
## 2️⃣ Encoding Categorical Features
```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])
```
## 3️⃣ Handling Missing Values
```python
df = df.dropna()
df = df.drop_duplicates()

```
## 4️⃣ Splitting Dataset
```python
from sklearn.model_selection import train_test_split
X = df.drop(['price'], axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

```
## 5️⃣ Feature Scaling
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

```
