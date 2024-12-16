import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, BoundaryNorm

# 定义Izhikevich模型
def izhikevich(V, u, I, a, b, c, d):
    """
    Izhikevich神经元模型的动态方程
    V: 膜电位
    u: 恢复变量
    I: 外部输入
    a, b, c, d: Izhikevich模型的参数
    """
    dV = 0.04 * V**2 + 5 * V + 140 - u + I  # 膜电位的更新方程
    du = a * (b * V - u)  # 恢复变量的更新方程

    V += dV
    u += du

    # 添加复位机制
    if V >= 30:  # 当膜电位超过阈值
        V = c     # 复位膜电位
        u += d    # 恢复变量增加

    return V, u

# 创建三维坐标，限制最小距离并扩大空间
N = 1000  # 节点数量
k = 4    # 每个节点的固定出向边数量
space_size = 10  # 三维空间大小
min_distance = 0.5  # 最小距离
np.random.seed(42)

# 生成三维坐标，确保最小距离
positions = []
while len(positions) < N:
    point = np.random.uniform(0, space_size, 3)
    if all(np.linalg.norm(point - np.array(p)) >= min_distance for p in positions):
        positions.append(point)
positions = np.array(positions)

# 创建有向图并添加边
G = nx.DiGraph()
G.add_nodes_from(range(N))

# 计算所有点之间的距离
distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
np.fill_diagonal(distances, np.inf)  # 自身距离设为无穷大，防止自连接

# 为每个点选择距离最近的 k 个点作为目标
for i in range(N):
    neighbors = np.argsort(distances[i])[:k]  # 最近的 k 个点
    for j in neighbors:
        G.add_edge(i, j)

# 初始化膜电位和恢复变量
V = np.full(N, -65.0, dtype=np.float64)  # 初始膜电位为-65 mV
V[0] = 60
u = np.full(N, 0.0, dtype=np.float64)    # 初始恢复变量为0

# 随机化每个神经元的模型参数
params = {
    "a": np.random.uniform(0.01, 0.03, N),
    "b": np.random.uniform(0.1, 0.3, N),
    "c": np.random.uniform(-68, -58, N),
    "d": np.random.uniform(2, 10, N),
}

# 设置随机输入，初始仅对一个节点施加强刺激
I = np.zeros(N)

# 创建传播权重矩阵（固定随机权重）
W = np.zeros((N, N))
for i, j in G.edges():
    W[i, j] = np.random.uniform(0.8, 1.2)  # 权重在 0.8 到 1.2 之间波动

# 动态绘制的设置
T = 200  # 模拟的时间步数

# 提取节点的三维坐标
xs, ys, zs = positions[:, 0], positions[:, 1], positions[:, 2]

# 自定义颜色映射
colors_under_70 = plt.cm.inferno(np.linspace(0, 0.5, 128))  # -80到-70灰色渐变
colors_70_to_60 = plt.cm.plasma(np.linspace(0.4, 0.9, 128)) #-70到-50
colors_60_to_0 = plt.cm.cividis(np.linspace(0.6, 1.0, 128)) #-50到0冷色渐变
colors_0_to_80 = plt.cm.inferno(np.linspace(0.9, 1.0, 256))  # 50到80暖色渐变
all_colors = np.vstack((colors_under_70,colors_70_to_60 ,colors_60_to_0, colors_0_to_80))
custom_cmap = ListedColormap(all_colors)

# 设置颜色的边界和分段
boundaries = [-80,-70,-65,-60,-55,-50,-40,-20,0,10,20,30,45,80]  # 边界递增
norm = BoundaryNorm(boundaries, custom_cmap.N, clip=True)

# 创建图形和3D子图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 初始化节点和边的绘图元素
sc = ax.scatter(xs, ys, zs, c=V, s=50, cmap=custom_cmap, norm=norm)

# 绘制有向边
lines = []
for i, j in G.edges():
    x_vals = [positions[i, 0], positions[j, 0]]
    y_vals = [positions[i, 1], positions[j, 1]]
    z_vals = [positions[i, 2], positions[j, 2]]
    line, = ax.plot(x_vals, y_vals, z_vals, color='gray', alpha=0.5, linestyle='-', linewidth=0.8)
    lines.append(line)

# 添加颜色条
cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=custom_cmap), ax=ax, shrink=0.6)
cbar.set_label("Membrane Potential (mV)")
cbar.set_ticks(boundaries)  # 非线性分布刻度

# 更新函数
def update(t):
    global V, u, I

    # 动态更新每个节点的输入电流
    new_I = np.zeros(N)
    for i in range(N):
        if V[i] > 0:  # 仅在膜电位大于0时向邻居传播刺激
            neighbors = list(G.successors(i))  # 获取出向邻居
            for j in neighbors:
                new_I[j] += W[i, j] * (V[i]*2 - params["c"][i])  # 根据膜电位生成刺激

    # 更新输入电流并添加随机噪声
    I = new_I + np.random.uniform(-0.5, 0.5, N)

    # 更新膜电位和恢复变量
    for i in range(N):
        V[i], u[i] = izhikevich(V[i], u[i], I[i], params["a"][i], params["b"][i], params["c"][i], params["d"][i])
        V[i] = np.clip(V[i], -80, 80)  # 限制膜电位的范围

    # 更新节点颜色
    sc.set_array(V)

    ax.set_title(f"Time Step: {t}")
    return sc, *lines

# 创建动画
ani = FuncAnimation(fig, update, frames=T, interval=200, blit=False)  # interval设置为200ms，放慢演化速度

# 保存动画为GIF文件
ani.save("izhikevich_network_modified_connections.gif", writer="pillow")

# 显示动画
plt.show()
