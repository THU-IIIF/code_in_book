# -*- coding: utf-8 -*-
# 文件名：svm_soft_margin_financial.py
# 演示软间隔 SVM 在财务违规识别任务中的应用
# 图中红色×点表示违规公司，蓝色○点表示未违规公司

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

plt.rcParams['font.family'] = 'Noto Serif CJK JP'
# ==============================
# 1. 模拟财务违规数据
# ==============================
# 特征如：资产负债率、应收账款率等，目标变量为是否违规（1: Yes, 0: No）
X, y = make_classification(
    n_samples=300,
    n_features=2,       # 两个主要财务指标
    n_redundant=0,
    n_clusters_per_class=1,
    class_sep=0.8,      # 类间分离度
    flip_y=0.1,         # 10%标签噪声
    random_state=42
)

# 标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练与测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# ==============================
# 2. 训练软间隔 SVM 模型
# ==============================
# C 越大 → 对误分类惩罚更强（间隔变窄）
svm_soft = svm.SVC(C=1.0, kernel='linear')
svm_soft.fit(X_train, y_train)

# ==============================
# 3. 模型性能评估
# ==============================
y_pred = svm_soft.predict(X_test)
print("分类报告：")
print(classification_report(y_test, y_pred))
print("混淆矩阵：")
print(confusion_matrix(y_test, y_pred))

# ==============================
# 4. 绘制软间隔分类边界
# ==============================
w = svm_soft.coef_[0]
b = svm_soft.intercept_[0]

# 生成网格以绘制决策边界
xx = np.linspace(-3, 3, 100)
yy = np.linspace(-3, 3, 100)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svm_soft.decision_function(xy).reshape(XX.shape)

# 绘制分类间隔线
plt.figure(figsize=(7, 6))
plt.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1],
            alpha=0.5, linestyles=['--', '-', '--'])

# 绘制训练样本点
# 非违规样本（y=0）用蓝色圆圈表示
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
            color='blue', marker='o', s=50, label='未违规公司')

# 违规样本（y=1）用红色叉号表示
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
            color='red', marker='x', s=60, label='违规公司')

# 标注支持向量（用黑色边框圈出）
plt.scatter(svm_soft.support_vectors_[:, 0],
            svm_soft.support_vectors_[:, 1],
            s=100, facecolors='none', edgecolors='black', linewidths=1.5,
            label='支持向量')

plt.title("软间隔 SVM 在财务违规识别中的分类边界", fontsize=13)
plt.xlabel("特征1：资产负债率（标准化）")
plt.ylabel("特征2：应收账款率（标准化）")
plt.legend()
plt.grid(alpha=0.2)
plt.savefig('svm_soft_margin_financial.png')