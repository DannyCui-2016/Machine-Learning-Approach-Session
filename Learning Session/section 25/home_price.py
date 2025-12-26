import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. 准备数据
# 房屋面积（平方米）
X = np.array([30, 40, 50, 60, 70, 80, 90]).reshape(-1, 1)

# 房价（单位：万元）
y = np.array([45, 60, 75, 90, 110, 130, 150])

# 2. 创建并训练模型
model = LinearRegression()
model.fit(X, y)

# 3. 预测新数据
new_area = np.array([[65]])
predicted_price = model.predict(new_area)

print(f"预测 {new_area[0][0]} 平方米的房价是：{predicted_price[0]:.2f} 万元")

# 4. 可视化结果
plt.scatter(X, y, color="blue", label="真实数据")
plt.plot(X, model.predict(X), color="red", label="模型预测")
plt.scatter(new_area, predicted_price, color="green", s=100, label="新预测点")

plt.xlabel("房屋面积 (㎡)")
plt.ylabel("房价 (万元)")
plt.title("房价预测 - 线性回归 Demo")
plt.legend()
plt.show()
plt.savefig("house_price_prediction.png")