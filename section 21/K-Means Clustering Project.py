import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ---------- 1️⃣ 模拟奥克兰房价与面积数据 ----------
# 单位：价格(万纽币)，面积(平方米)
# 生成三种典型区域的数据
np.random.seed(42)

# 高端区（Remuera, Parnell 等）
high_end = np.random.normal(loc=[180, 250], scale=[20, 30], size=(50, 2))

# 中等区（Mt Eden, Newmarket 等）
mid_range = np.random.normal(loc=[100, 160], scale=[15, 25], size=(50, 2))

# 平价区（Papatoetoe, Henderson 等）
low_end = np.random.normal(loc=[60, 110], scale=[10, 20], size=(50, 2))

# 合并数据
data = np.vstack((high_end, mid_range, low_end))

# ---------- 2️⃣ 建立并训练 K-Means 模型 ----------
k = 3  # 聚成3类
kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(data)

labels = kmeans.labels_
centers = kmeans.cluster_centers_

# ---------- 3️⃣ 可视化聚类结果 ----------
plt.figure(figsize=(8,6))
colors = ['#2ecc71', '#f1c40f', '#e74c3c']  # 三种颜色代表三类区域

for i in range(k):
    cluster_points = data[labels == i]
    plt.scatter(cluster_points[:,0], cluster_points[:,1],
                c=colors[i], label=f'Cluster {i+1}', alpha=0.7)

# 绘制聚类中心
plt.scatter(centers[:,0], centers[:,1], c='black', marker='X', s=200, label='Centers')

plt.title("Auckland Housing Segmentation (K-Means)")
plt.xlabel("House Price (10k NZD)")
plt.ylabel("House Size (sqm)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()
plt.savefig("auckland_housing_pricing_kmeans.png")

# ---------- 4️⃣ 打印聚类中心 ----------
for i, c in enumerate(centers, 1):
    print(f"Cluster {i}: 平均价格 {c[0]:.1f} 万纽币, 平均面积 {c[1]:.1f} 平方米")
