# ----------------------------
# 1. 导入所需库
# ----------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.decomposition import PCA  # 用于降维可视化
from sklearn.metrics import accuracy_score, adjusted_rand_score
from sklearn.model_selection import train_test_split

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ----------------------------
# 2. 数据加载与基本信息
# ----------------------------
print("===== 数据加载 =====")
iris = load_iris()
X = iris.data  # 特征（4维）
y_true = iris.target  # 真实标签（0,1,2三类）
feature_names = iris.feature_names
target_names = iris.target_names

print(f"数据集特征：{feature_names}")
print(f"数据集类别：{target_names}")
print(f"样本数量：{X.shape[0]}，特征维度：{X.shape[1]}")


# ----------------------------
# 3. K-means聚类实战与评估
# ----------------------------
print("\n===== K-means聚类 =====")

# 3.1 聚类实现（K=3）
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X)

# 3.2 聚类结果可视化（PCA降维到2D）
pca = PCA(n_components=2)  # 4维→2维，方便可视化
X_pca = pca.fit_transform(X)

plt.figure(figsize=(12, 5))

# 子图1：真实标签分布
plt.subplot(1, 2, 1)
sc1 = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='viridis', s=50)
plt.title('真实标签分布', fontsize=12)
plt.xlabel('PCA维度1')
plt.ylabel('PCA维度2')
plt.colorbar(sc1, ticks=[0, 1, 2], label='真实类别')

# 子图2：K-means聚类结果
plt.subplot(1, 2, 2)
sc2 = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='plasma', s=50)
plt.title('K-means聚类结果（K=3）', fontsize=12)
plt.xlabel('PCA维度1')
plt.ylabel('PCA维度2')
plt.colorbar(sc2, ticks=[0, 1, 2], label='聚类标签')

plt.tight_layout()
plt.show()

# 3.3 肘部法则选择最佳K值（K=1~10）
sse = []
k_range = range(1, 11)
for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42)
    kmeans_temp.fit(X)
    sse.append(kmeans_temp.inertia_)  # 记录SSE

# 绘制肘部曲线
plt.figure(figsize=(8, 4))
plt.plot(k_range, sse, 'o-', color='darkblue', linewidth=2)
plt.xlabel('聚类数量K')
plt.ylabel('误差平方和（SSE）')
plt.title('肘部法则曲线（最佳K值选择）')
plt.grid(alpha=0.3)
plt.scatter(3, sse[2], color='red', s=150, zorder=5)  # 标记肘部
plt.annotate('肘部（K=3）', xy=(3, sse[2]), xytext=(4, sse[2]+10),
             arrowprops=dict(arrowstyle='->', color='red'))
plt.show()

# 3.4 聚类结果评估（调整兰德指数ARI）
ari_kmeans = adjusted_rand_score(y_true, y_kmeans)
print(f"K-means与真实标签的调整兰德指数（ARI）：{ari_kmeans:.4f}")


# ----------------------------
# 4. SVM分类实战与参数调优
# ----------------------------
print("\n===== SVM分类 =====")

# 4.1 数据划分（训练集80%，测试集20%）
X_train, X_test, y_train, y_test = train_test_split(
    X, y_true, test_size=0.2, random_state=42, stratify=y_true  # 保持类别分布
)
print(f"训练集样本数：{X_train.shape[0]}，测试集样本数：{X_test.shape[0]}")

# 4.2 线性SVM（kernel='linear'）
svm_linear = SVC(kernel='linear', random_state=42)
svm_linear.fit(X_train, y_train)
y_pred_linear = svm_linear.predict(X_test)
acc_linear = accuracy_score(y_test, y_pred_linear)
print(f"线性SVM测试集准确率：{acc_linear:.4f}")

# 4.3 非线性SVM（RBF核）与参数调优
C_list = [0.1, 1, 10, 100]  # 惩罚系数
gamma_list = [0.01, 0.1, 1, 10]  # 核系数
best_acc = 0
best_params = {}

print("\nRBF核SVM参数调优（C, gamma）与准确率：")
for C in C_list:
    for gamma in gamma_list:
        svm_rbf = SVC(kernel='rbf', C=C, gamma=gamma, random_state=42)
        svm_rbf.fit(X_train, y_train)
        acc = accuracy_score(y_test, svm_rbf.predict(X_test))
        print(f"C={C}, gamma={gamma} → 准确率：{acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            best_params = {'C': C, 'gamma': gamma}

print(f"\n最优RBF核参数：{best_params}，最高准确率：{best_acc:.4f}")

# 4.4 SVM分类边界可视化（PCA降维后）
# 用PCA降维训练集和测试集到2D
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 用最优参数训练RBF核SVM
svm_best = SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'], random_state=42)
svm_best.fit(X_train_pca, y_train)

# 生成网格点绘制边界
h = 0.02  # 网格步长
x_min, x_max = X_train_pca[:, 0].min()-1, X_train_pca[:, 0].max()+1
y_min, y_max = X_train_pca[:, 1].min()-1, X_train_pca[:, 1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# 预测网格点类别
Z = svm_best.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制边界与样本
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.3)  # 分类区域
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='coolwarm',
            edgecolors='black', s=60, label='训练集')  # 训练样本
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap='coolwarm',
            edgecolors='black', marker='^', s=80, label='测试集')  # 测试样本
plt.xlabel('PCA维度1')
plt.ylabel('PCA维度2')
plt.title(f'RBF核SVM分类边界（C={best_params["C"]}, gamma={best_params["gamma"]}）')
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# ----------------------------
# 5. 两种范式对比分析
# ----------------------------
print("\n===== 模型对比分析 =====")
print("| 模型         | 学习范式 | 依赖标签 | 评估指标       | 指标值   |")
print("|--------------|----------|----------|----------------|----------|")
print(f"| K-means      | 无监督   | 否       | 调整兰德指数   | {ari_kmeans:.4f} |")
print(f"| 线性SVM      | 有监督   | 是       | 测试集准确率   | {acc_linear:.4f} |")
print(f"| RBF核SVM     | 有监督   | 是       | 测试集准确率   | {best_acc:.4f} |")

print("\n结论：有监督学习（SVM）利用标签信息，分类准确率显著高于无监督聚类（K-means）；"
      "核技巧可增强SVM处理非线性数据的能力，参数调优对性能有重要影响。")