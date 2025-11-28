# Mastercard Stock
![mastercard-stock](/mastercard.jpg)


## Procedures
- Import the libraries
    - pandas
    - matplotlib
    - scikit-learn
    - seaborn
    - numpy
    - yfinance 
- Data Acquisition and Inital Setup
    - Data acquired from the Yahoo Finance API
- Feature Engineering
    - Feature: Close, SMA_5, SMA_20, Daily_Return, Volatility_10, Price_Volume_Ratio
    - Target: Price_Up
- Pre-Training Visualization

![pre-training-visualization](/output.png)
- Data Splitting
    - Split the data into training and testing sets (80% train, 20% test)
    - shuffle=False is cruical for time-series data into maintain temproral order
- Data Scaling
    - Initialize the StandardScaler 
    - StandardScaler standardizes features by removing the mean and scaling to unit variance
    - Fit the scaler only on the training data to prevent data leakage from the test set
    - Apply the fitted scaler to transform both training and testing sets
- Model Comparison
    - Logistic Regression
    - K-Nearest Neighbors
    - Random Forest
    - Support Vector Machine (Linear)
    - Gaussian Naive Bayes
- Hyperparameter Tuning
    - Using GridSearchCV(estimator, param_grid, cv, scoring, verbose, n_jobs)
- Post-Training Visualization
- Prediction Function for new data

## Tech Stack and Tools
- Programming language
    - Python 
- libraries
    - scikit-learn
    - pandas
    - numpy
    - seaborn
    - matplotlib
    - yfinance
- Environment
    - Jupyter Notebook
    - Anaconda
- IDE
    - VSCode

You can install all dependencies via:
```
pip install -r requirements.txt
```

## Usage Instructions
To run this project locally:
1. Clone the repository:
```
git clone https://github.com/charlesakinnurun/mastercard-stock.git
cd mastercard-stock
```
2. Install required packages
```
pip install -r requirements.txt
```
3. Open the notebook:
```
jupyter notebook model.ipynb

```


## Project Structure
```
mastercard-stock/
│
├── model.ipynb  
|── model.py    
|── mastercard_stock_data.csv  
├── requirements.txt 
├── mastercard.jpg       
├── output.png              
├── SECURITY.md        
├── CONTRIBUTING.md    
├── CODE_OF_CONDUCT.md 
├── LICENSE
└── README.md          

```

