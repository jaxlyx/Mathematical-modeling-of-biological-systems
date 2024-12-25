import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, BoundaryNorm

# 定义Izhikevich模型
def izhikevich(V, u, I, a, b, c, d):
    dV = 0.04 * V**2 + 5 * V + 140 - u + I
    du = a * (b * V - u)
    V += dV
    u += du
    if V >= 30:  # Reset机制
        V = c
        u += d
    return V, u

# 定义神经元类
class Neuron:
    def __init__(self, index, position, a, b, c, d):
        self.index = index
        self.position = position
        self.V = -65.0  # 初始膜电位
        self.u = 0.0    # 初始恢复变量
        self.a = a      # 参数a
        self.b = b      # 参数b
        self.c = c      # 参数c
        self.d = d      # 参数d
        self.I = 0      # 初始输入电流
        self.I_ext = 0  # 外部刺激电流（新增）
        self.neighbors = []  # 邻居神经元
        self.weights = []  # 每个邻居对应的突触权重

    def update(self, I):
        self.I = I + self.I_ext  # 外部刺激加入到神经元的输入中
        self.V, self.u = izhikevich(self.V, self.u, self.I, self.a, self.b, self.c, self.d)
        self.V = np.clip(self.V, -80, 80)  # 限制膜电位

    def add_neighbors(self, neighbors, weights):
        self.neighbors = neighbors
        self.weights = weights

    def apply_external_stimulus(self, stimulus):
        self.I_ext = stimulus  # 设置外部刺激


class Network:
    def __init__(self, N, k, space_size, min_distance, center_point, connection_prob_func=None, weight_func=None):
        """
        初始化神经网络

        :param N: 神经元数量
        :param k: 每个神经元的出向边数量
        :param space_size: 空间的大小
        :param min_distance: 神经元之间的最小距离
        :param center_point: 高斯分布的中心点（三维坐标）
        :param connection_prob_func: 连接概率函数（可选）
        :param weight_func: 权重函数（可选）
        """
        self.N = N
        self.k = k
        self.space_size = space_size
        self.min_distance = min_distance
        self.center_point = center_point  # 使用传入的 center_point
        self.connection_prob_func = connection_prob_func if connection_prob_func is not None else self.default_connection_prob
        self.weight_func = weight_func if weight_func is not None else self.default_weight_func
        
        self.neurons = []
        self.positions = []
        self.G = nx.DiGraph()
        
        self.create_network()

    def default_connection_prob(self, distance, max_distance=10, base_prob=0.1):
        """默认的连接概率函数，基于神经元间的距离"""
        if distance > max_distance:
            return 0  # 超过最大距离不连接
        return base_prob * np.exp(-distance**2 / (2 * (max_distance / 2)**2))

    def default_weight_func(self, distance, max_distance=10, base_weight=1.0):
        """默认的权重函数，基于距离的反比"""
        if distance > max_distance:
            return 0  # 超过最大距离不传递信号
        return base_weight * np.exp(-distance**2 / (2 * (max_distance / 2)**2))

    def create_network(self):
        np.random.seed(42)
        
        # 使用高斯分布生成神经元位置
        positions = []
        while len(positions) < self.N:
            # 使用高斯分布围绕中心点生成位置
            point = np.random.normal(loc=self.center_point, scale=self.space_size / 6, size=3)
            # 限制点在空间范围内
            point = np.clip(point, 0, self.space_size)
            if all(np.linalg.norm(point - np.array(p)) >= self.min_distance for p in positions):
                positions.append(point)
        self.positions = np.array(positions)
        
        # 随机化Izhikevich模型的参数
        params = {
            "a": np.random.uniform(0.01, 0.03, self.N),
            "b": np.random.uniform(0.1, 0.3, self.N),
            "c": np.random.uniform(-68, -58, self.N),
            "d": np.random.uniform(2, 10, self.N),
        }
        
        # 创建神经元
        self.neurons = [Neuron(i, self.positions[i], params["a"][i], params["b"][i], params["c"][i], params["d"][i]) for i in range(self.N)]
        
        # 创建图
        self.G.add_nodes_from(range(self.N))
        
        # 计算所有神经元间的距离
        distances = np.linalg.norm(self.positions[:, None, :] - self.positions[None, :, :], axis=-1)
        np.fill_diagonal(distances, np.inf)  # 防止自连接

        # 初始化连接
        for i in range(self.N):
            neighbors = []
            weights = []
            for j in range(self.N):
                if i != j:
                    prob = self.connection_prob_func(distances[i, j])
                    if np.random.rand() < prob:  # 判断是否连接
                        neighbors.append(j)
                        weights.append(self.weight_func(distances[i, j]))
            self.neurons[i].add_neighbors(neighbors, weights)
            # 添加连接到图中
            for j in neighbors:
                self.G.add_edge(i, j)

    def add_external_stimuli(self, stimulus_func):
        """
        为神经元添加外部刺激
        :param stimulus_func: 外部刺激函数，根据神经元的索引或位置生成刺激值
        """
        for i, neuron in enumerate(self.neurons):
            neuron.apply_external_stimulus(stimulus_func(i))

# 例如，创建一个简单的外部刺激函数（可以根据需要自定义）
def external_stimulus(i):
    """一个简单的外部刺激函数：给一些神经元施加刺激"""
    if i % 10 == 0:  # 每10个神经元施加一次外部刺激
        return 10  # 施加一个10mV的刺激
    return 0

# 设置颜色映射
def setup_color_map():
    colors_under_70 = plt.cm.inferno(np.linspace(0, 0.5, 128))
    colors_70_to_60 = plt.cm.plasma(np.linspace(0.4, 0.9, 128))
    colors_60_to_0 = plt.cm.cividis(np.linspace(0.6, 1.0, 128))
    colors_0_to_80 = plt.cm.inferno(np.linspace(0.9, 1.0, 256))
    all_colors = np.vstack((colors_under_70, colors_70_to_60, colors_60_to_0, colors_0_to_80))
    custom_cmap = ListedColormap(all_colors)

    boundaries = [-80,-70,-65,-60,-55,-50,-40,-20,0,10,20,30,45,80]
    norm = BoundaryNorm(boundaries, custom_cmap.N, clip=True)
    
    return custom_cmap, norm, boundaries

# 动画更新函数
def update(t, neurons, sc, ax, lines, positions, custom_cmap, norm, T):
    new_I = np.zeros(len(neurons))
    for i, neuron in enumerate(neurons):
        if neuron.V > 0:
            for j, weight in zip(neuron.neighbors, neuron.weights):
                new_I[j] += weight * (neuron.V * 2 - neuron.c)

    for i, neuron in enumerate(neurons):
        neuron.update(new_I[i])

    sc.set_array([neuron.V for neuron in neurons])
    ax.set_title(f"Time Step: {t}")
    return sc, *lines

# 生成动画
def create_animation(neurons, positions, G, T, custom_cmap, norm, boundaries, filename):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    xs, ys, zs = positions[:, 0], positions[:, 1], positions[:, 2]

    sc = ax.scatter(xs, ys, zs, c=[neuron.V for neuron in neurons], s=50, cmap=custom_cmap, norm=norm)

    lines = []
    for i, j in G.edges():
        x_vals = [positions[i, 0], positions[j, 0]]
        y_vals = [positions[i, 1], positions[j, 1]]
        z_vals = [positions[i, 2], positions[j, 2]]
        line, = ax.plot(x_vals, y_vals, z_vals, c='gray', alpha=0.2)
        lines.append(line)

    ani = FuncAnimation(fig, update, frames=T, fargs=(neurons, sc, ax, lines, positions, custom_cmap, norm, boundaries), interval=50)

    ani.save(filename, writer='imagemagick', fps=10)

# 设置参数
N = 50  # 每个网络的节点数量
k = 4   # 每个节点的固定出向边数量
space_size = 10
min_distance = 0.3
T = 100  # 时间步数

# 创建网络
network1 = Network(N, k, space_size, min_distance, [4, 4, 4])
network2 = Network(N, k, space_size, min_distance, [8, 8, 8])

# 合并两个网络
neurons = network1.neurons + network2.neurons
positions = np.vstack([network1.positions, network2.positions])
G = nx.disjoint_union(network1.G, network2.G)

# 添加外部刺激
network1.add_external_stimuli(external_stimulus)
network2.add_external_stimuli(external_stimulus)

# 设置颜色映射
custom_cmap, norm, boundaries = setup_color_map()

# 生成动画
create_animation(neurons, positions, G, T, custom_cmap, norm, boundaries, "combined_network_with_stimulus.gif")

plt.show()