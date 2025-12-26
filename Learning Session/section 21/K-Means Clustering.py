# kmeans_demo.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# 1️⃣ 生成模拟数据
X, y_true = make_blobs(
    n_samples=300,   # 样本数量
    centers=3,       # 真实的簇数量
    cluster_std=0.60,# 每个簇的方差（越大越散）
    random_state=0
)

# 2️⃣ 画出原始数据（没有聚类前）
plt.scatter(X[:, 0], X[:, 1], s=30)
plt.title("Original Data (Unlabeled)")
plt.show()

# 3️⃣ 创建 KMeans 模型并训练
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# 4️⃣ 可视化聚类结果
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=30, cmap='viridis')

# 画出聚类中心
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.7, marker='X')

plt.title("K-Means Clustering Result (3 clusters)")
plt.show()
plt.savefig("kmeans_clustering_result.png")

# 5️⃣ 打印结果信息
print("聚类中心坐标：\n", centers)
print("每个样本所属簇：\n", y_kmeans[:10])
print("真实标签（前10个样本）：\n", y_true[:10])