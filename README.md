# SALES-ANALYSIS

# 📊 Sales Analysis Dashboard — Profit Prediction using Machine Learning

An end-to-end **Data Analysis & Machine Learning** project that explores a retail sales transaction dataset and builds a **Random Forest Regressor** to predict order-level **Profit**.

> Individual AIML Project by **M.S.MUHAMMED RABBANEE**

---

## 📁 Project Overview

Businesses generate large volumes of transactional sales data, but without systematic analysis it's hard to know which orders, products, or regions drive profitability — and which erode it.

This project analyzes a sales dataset end-to-end:
- Cleans and explores the data (EDA)
- Detects and removes outliers
- Engineers new business-relevant features
- Encodes and scales the data
- Trains and evaluates a regression model to predict **Profit**
- Saves the model for reuse/deployment

---

## 🗂️ Dataset

**File:** `sales_analysis_dataset.csv`

| Category | Columns |
|---|---|
| Identifiers | `Order_ID`, `Order_Date`, `Ship_Date`, `Customer_Name` |
| Location & Product | `Segment`, `Region`, `State`, `Category`, `Product` |
| Order Details | `Quantity`, `Unit_Price`, `Discount_Percent`, `Discount_Amount` |
| Financials | `Sales`, `Cost`, `Profit` (**Target**) |
| Logistics | `Ship_Mode`, `Payment_Mode` |

**Target Variable:** `Profit` (continuous → Regression task)

---

## 🛠️ Technologies Used

- Python 3
- Google Colab
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

---

## 🔄 Machine Learning Workflow

```
Load Dataset → Data Cleaning & EDA → Outlier Removal (IQR)
→ Feature Engineering → Feature Selection → Encoding
→ Feature Scaling → Train-Test Split → Random Forest
→ Model Evaluation → Save Model → Deployment
```

---

## 📌 Project Steps

### 1. Exploratory Data Analysis (EDA)
- Import libraries, load & display dataset
- `head()`, `tail()`, `shape`, `columns`, `dtypes`, `info()`, `describe()`
- Check missing values, duplicates & unique values
- Target variable distribution
- Correlation matrix & heatmap
- Histograms, count plots, box plots, scatter plot, pair plot, KDE plot, violin plot, bar plot, pie chart

### 2. Data Preprocessing & Feature Engineering
- **Outlier Detection:** IQR method (1.5×IQR rule) on `Sales`, `Cost`, `Profit`, `Quantity`, etc.
- **Feature Engineering:**
  - `Order_Year`, `Order_Month`, `Order_Day`, `Order_Weekday`
  - `Shipping_Days` = Ship_Date − Order_Date
  - `Profit_Margin` = Profit / Sales
  - `Revenue_Per_Unit` = Sales / Quantity
- **Feature Selection:** Dropped identifier columns (`Order_ID`, `Order_Date`, `Ship_Date`, `Customer_Name`)
- **Encoding:** One-Hot Encoding on `Segment`, `Region`, `State`, `Category`, `Product`, `Ship_Mode`, `Payment_Mode`
- **Feature Scaling:** `StandardScaler` on numeric features

### 3. Model Building
- **Train-Test Split:** 80/20
- **Algorithm:** `RandomForestRegressor` (n_estimators=200, max_depth=10)
- **Evaluation Metrics:** MAE, RMSE, R² Score
- Feature importance visualization

### 4. Model Persistence & Deployment
- Model saved using `joblib` (`sales_profit_model.pkl`)
- Scaler and column list saved for consistent inference
- Reusable `predict_profit()` function for new order data

---

## 📈 Results

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error of predicted profit |
| RMSE | Root Mean Squared Error |
| R² Score | Variance in Profit explained by the model |

Random Forest was chosen for its ability to handle mixed numeric/categorical features, capture non-linear relationships, and resist overfitting via ensemble bagging.

---

## 🚀 How to Run

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `sales_analysis_dataset.csv` and the notebook
3. Run all cells in order — Import Libraries → EDA → Preprocessing → Model Training → Evaluation → Save Model
4. Download the generated `.pkl` files for reuse

```bash
# Local setup (optional, instead of Colab)
pip install pandas numpy matplotlib seaborn scikit-learn joblib
python sales_analysis.py
```

---

## 📂 Repository Structure

```
├── sales_analysis_dataset.csv        # Raw dataset
├── Sales_Analysis_Dashboard.ipynb    # Main Colab/Jupyter notebook
├── sales_profit_model.pkl            # Saved trained model
├── scaler.pkl                        # Saved StandardScaler
├── model_columns.pkl                 # Saved encoded column order
└── README.md                         # Project documentation
```

---

## 🔮 Future Enhancements

- Deep learning models for improved accuracy
- Real-time sales prediction pipeline
- Web dashboard deployment (Streamlit/Flask)
- Cloud integration & mobile app
- Live data pipeline with automatic retraining

---

## 👤 Author

**M.S.MUHAMMED RABBANEE**
Individual AIML Project — Sales Analysis Dashboard

---

## 📄 License

This project is open-source and available for educational purposes.
