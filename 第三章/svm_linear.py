# ============================================
# 线性可分支持向量机（SVM）在财务违规识别中的演示
# 作者：刘彦超
# 日期：2025
# ============================================

# 导入必要库
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#import matplotlib

#a=sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])

#for i in a:
#    print(i)

plt.rcParams['font.family'] = 'Noto Serif CJK JP'
# -----------------------------
# 1. 数据模拟
# -----------------------------
# 假设我们有两个核心财务指标：
# X1 = 应计利润率（越高越可能违规）
# X2 = 经营现金流与净利润比（越低越可能违规）

np.random.seed(42)

# 合规样本（非违规，y=-1）
n_good = 80
x1_good = np.random.normal(loc=0.2, scale=0.1, size=n_good)
x2_good = np.random.normal(loc=0.6, scale=0.1, size=n_good)

# 违规样本（y=+1）
n_bad = 40
x1_bad = np.random.normal(loc=0.5, scale=0.1, size=n_bad)
x2_bad = np.random.normal(loc=0.3, scale=0.1, size=n_bad)

# 拼接数据
X = np.vstack((np.column_stack((x1_good, x2_good)),
               np.column_stack((x1_bad, x2_bad))))
y = np.hstack(([-1]*n_good, [1]*n_bad))

# -----------------------------
# 2. 数据标准化
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 拆分训练与测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# -----------------------------
# 3. 训练线性 SVM 模型
# -----------------------------
# kernel='linear' 代表使用线性核函数
# C=1.0 为惩罚系数，越大越趋向“硬间隔”
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X_train, y_train)

# -----------------------------
# 4. 模型结果分析
# -----------------------------
print("支持向量数量：", len(clf.support_))
print("支持向量索引：", clf.support_)
print("模型系数（权重 w）：", clf.coef_)
print("截距 b：", clf.intercept_)
print("\n分类报告（测试集）：")
print(classification_report(y_test, clf.predict(X_test)))

# 计算间隔（margin = 2 / ||w||）
margin = 2 / np.linalg.norm(clf.coef_)
print(f"\n模型间隔（margin） ≈ {margin:.4f}")

# -----------------------------
# 5. 可视化决策边界与支持向量
# -----------------------------
plt.figure(figsize=(8,6))

# 恢复为原尺度方便解释
X_vis = scaler.inverse_transform(X_train)
x1_min, x1_max = X_vis[:,0].min() - 0.1, X_vis[:,0].max() + 0.1
x2_min, x2_max = X_vis[:,1].min() - 0.1, X_vis[:,1].max() + 0.1

# 绘制决策面
xx, yy = np.meshgrid(np.linspace(x1_min, x1_max, 200),
                     np.linspace(x2_min, x2_max, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = clf.decision_function(scaler.transform(grid)).reshape(xx.shape)

plt.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['blue', 'black', 'red'],
            linestyles=['--', '-', '--'])

# 绘制样本点
plt.scatter(x1_good, x2_good, c='skyblue', label='合规 (y=-1)')
plt.scatter(x1_bad, x2_bad, c='orange', label='违规 (y=+1)', marker='x')

# 绘制支持向量（空心大圈）
sv = scaler.inverse_transform(clf.support_vectors_)
plt.scatter(sv[:, 0], sv[:, 1], s=150, facecolors='none', edgecolors='k', label='支持向量')

plt.xlabel('应计利润率')
plt.ylabel('现金流/净利润比')
plt.title('线性SVM在财务违规识别中的决策边界示意')
plt.legend()
plt.grid(True)
plt.savefig('svm_linear.png')
