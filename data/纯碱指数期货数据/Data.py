import pandas as pd
import numpy as np
import os
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time

folder_path = r'C:\Users\liumr\Desktop\715\index'

def load_and_merge_files(folder_path, file_names):
    data_frames = []
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_excel(file_path)
        data_frames.append(df)

file_names = [
    'TradingData.xlsx', '%B.xlsx', 'AH.xlsx', 'AMO.xlsx', 'B3612.xlsx', 'BB.xlsx', 'BBI.xlsx',
    'BBIBOLL.xlsx', 'BIAS.xlsx', 'BOLL.xlsx', 'BRAR.xlsx', 'CCI.xlsx',
    'CDP.xlsx', 'CHAIKIN.xlsx', 'CR.xlsx', 'DDI.xlsx', 'DMA.xlsx', 'DMI.xlsx',
    'EMV.xlsx', 'ENV.xlsx', 'ICHIMOKU.xlsx', 'KDJ.xlsx', 'KELT.xlsx', 'LWR.xlsx',
    'MACD.xlsx', 'MASS.xlsx', 'MIKE.xlsx', 'MTM.xlsx', 'OBV.xlsx', 'OI.xlsx',
    'OSC.xlsx', 'PB.xlsx', 'PE.xlsx', 'PSY.xlsx', 'ROC.xlsx', 'RSI.xlsx',
    'SAR.xlsx', 'SLOWKD.xlsx', 'SOBV.xlsx', 'StdDev.xlsx', 'TAPI.xlsx',
    'TRIX.xlsx', 'TURNOVER.xlsx', 'TWR.xlsx', 'VOL.xlsx',
    'VR.xlsx', 'W&R.xlsx', 'WVAD.xlsx', '市值.xlsx', '最大回撤率.xlsx', '资金流向.xlsx'
]

def load_and_merge_files(folder_path, file_names):
    data_frames = []
    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_excel(file_path)
        data_frames.append(df)
    
    merged_data = pd.concat(data_frames, axis=1, join='inner')
    merged_data = merged_data.loc[:, ~merged_data.columns.duplicated()]

    if '代码' in merged_data.columns:
        merged_data.drop(columns=['代码'], inplace=True)
    if '名称' in merged_data.columns:
        merged_data.drop(columns=['名称'], inplace=True)
    if 'NEGO' in merged_data.columns:
        merged_data.drop(columns=['NEGO'], inplace=True)
    if 'OI' in merged_data.columns:
        merged_data.drop(columns=['OI'], inplace=True)

    return merged_data

data = load_and_merge_files(folder_path, file_names)

if '日期' in data.columns:
    data.set_index('日期', inplace=True)


def convert_percentage_to_float(x):
    if isinstance(x, str) and '%' in x:
        return float(x.replace('%', '')) / 100
    return x

for col in data.columns:
    if data[col].dtype == 'object':
        
        data[col] = data[col].str.replace('亿', '', regex=False)
        
        data[col] = data[col].str.replace('万', '', regex=False)
        
        data[col] = data[col].apply(convert_percentage_to_float)
    
        data[col] = pd.to_numeric(data[col], errors='coerce')


data.fillna(method='ffill', inplace=True)
data.fillna(method='bfill', inplace=True)

scaler = StandardScaler()
X = scaler.fit_transform(data.drop(columns=['收盘价(元)']))

y = data['收盘价(元)'].values

# 构建训练和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

xgb_model = XGBRegressor()
rf_model = RandomForestRegressor()
dt_model = DecisionTreeRegressor()
lasso_model = Lasso(alpha=0.1, max_iter=10000)


# 训练随机森林模型
print("Training Random Forest model...")
start_time = time.time()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
end_time = time.time()
print(f"Random Forest training completed in {end_time - start_time:.2f} seconds.")
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
corr_rf = np.corrcoef(y_test, y_pred_rf)[0, 1]

print(f'Random Forest - MSE: {mse_rf}, RMSE: {rmse_rf}, MAE: {mae_rf}, R²: {r2_rf}, Corr: {corr_rf}')

# 训练决策树模型
print("Training Decision Tree model...")
start_time = time.time()
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
end_time = time.time()
print(f"Decision Tree training completed in {end_time - start_time:.2f} seconds.")
mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mse_dt)
mae_dt = mean_absolute_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)
corr_dt = np.corrcoef(y_test, y_pred_dt)[0, 1]

print(f'Decision Tree - MSE: {mse_dt}, RMSE: {rmse_dt}, MAE: {mae_dt}, R²: {r2_dt}, Corr: {corr_dt}')

# 训练Lasso模型
print("Training Lasso model...")
start_time = time.time()
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)
end_time = time.time()
print(f"Lasso training completed in {end_time - start_time:.2f} seconds.")
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mse_lasso)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)
corr_lasso = np.corrcoef(y_test, y_pred_lasso)[0, 1]

print(f'Lasso - MSE: {mse_lasso}, RMSE: {rmse_lasso}, MAE: {mae_lasso}, R²: {r2_lasso}, Corr: {corr_lasso}')

# 训练XGBoost模型
print("Training XGBoost model...")
start_time = time.time()
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
end_time = time.time()
print(f"XGBoost training completed in {end_time - start_time:.2f} seconds.")
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
corr_xgb = np.corrcoef(y_test, y_pred_xgb)[0, 1]

print(f'XGBoost - MSE: {mse_xgb}, RMSE: {rmse_xgb}, MAE: {mae_xgb}, R²: {r2_xgb}, Corr: {corr_xgb}')

# 训练随机森林模型
print("Training Random Forest model...")
start_time = time.time()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
end_time = time.time()
print(f"Random Forest training completed in {end_time - start_time:.2f} seconds.")
mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
corr_rf = np.corrcoef(y_test, y_pred_rf)[0, 1]
print(f'Random Forest - MSE: {mse_rf}, RMSE: {rmse_rf}, MAE: {mae_rf}, R²: {r2_rf}, Corr: {corr_rf}')

'''
print("Starting GridSearchCV for XGBoost...")
start_time = time.time()
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
end_time = time.time()
print(f"GridSearchCV completed in {end_time - start_time:.2f} seconds.")

best_xgb_model = grid_search.best_estimator_
best_xgb_model.fit(X_train, y_train)

y_pred_best_xgb = best_xgb_model.predict(X_test)

plt.figure()
plt.plot(y_test, label='Actual')
plt.plot(y_pred_best_xgb, label='Predicted (XGBoost)')
plt.legend()
plt.show()
'''


# 绘制结果图表
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual', color='black')
plt.plot(y_pred_xgb, label='Predicted (XGBoost)', linestyle='dashed')
plt.plot(y_pred_rf, label='Predicted (Random Forest)', linestyle='dotted')
plt.plot(y_pred_dt, label='Predicted (Decision Tree)', linestyle='dashdot')
plt.plot(y_pred_lasso, label='Predicted (Lasso)', linestyle='dashdot')
#plt.plot(y_pred_nn, label='Predicted (Neural Network)', linestyle='dotted')

plt.legend()
plt.title('Actual vs Predicted Values')
plt.xlabel('Sample Index')
plt.ylabel('Price')
plt.show()
