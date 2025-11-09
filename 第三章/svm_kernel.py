# -*- coding: utf-8 -*-
# 文件名：svm_kernel_rbf_financial.py
# 演示 RBF 核 SVM 在财务违规识别中的应用

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

plt.rcParams['font.family'] = 'Noto Serif CJK JP'
# 1. 构造模拟财务数据
X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_redundant=0,
    class_sep=0.5,
    n_clusters_per_class=1,
    flip_y=0.05,
    random_state=42
)

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 2. 使用 RBF 核训练 SVM
model_rbf = svm.SVC(C=1.0, kernel='rbf', gamma=0.8)
model_rbf.fit(X_train, y_train)

# 3. 绘制分类结果
xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
Z = model_rbf.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(7,6))
plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], linestyles=['--', '-', '--'], alpha=0.6)

# 绘制样本点
plt.scatter(X_train[y_train==0,0], X_train[y_train==0,1], c='blue', marker='o', label='未违规')
plt.scatter(X_train[y_train==1,0], X_train[y_train==1,1], c='red', marker='x', label='违规')
plt.scatter(model_rbf.support_vectors_[:,0], model_rbf.support_vectors_[:,1],
            s=80, facecolors='none', edgecolors='black', label='支持向量')

plt.title('RBF核SVM的非线性分类边界（财务违规识别）', fontsize=13)
plt.xlabel('特征1：资产负债率（标准化）')
plt.ylabel('特征2：应收账款率（标准化）')
plt.legend()
plt.grid(alpha=0.2)
plt.savefig('svm_kernel_rbf_financial.png', dpi=300)
