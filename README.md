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

### **📌 Sample Columns**
| symboling | fueltype | carbody | wheelbase | carwidth | horsepower | price |
|-----------|---------|--------|----------|---------|-----------|------|
| 3         | gas     | sedan  | 88.6     | 64.1    | 111       | 13495 |
| 2         | diesel  | hatchback | 94.5   | 65.2    | 154       | 16500 |

---

## **📌 Data Preprocessing**
### **1️⃣ Removing Unnecessary Columns**
```python
df = df.drop(['car_ID', 'CarName'], axis=1)
