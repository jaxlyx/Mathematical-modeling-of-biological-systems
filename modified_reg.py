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


# 创建多个独立的小世界网络
N_main = 15 # 主网络节点数量，可调节
N_subnets = 1 # 调节网络的数量
k = 3  # 每个节点的固定出向边数量
T = 800  # 模拟的时间步数

# 初始化图形对象
G = nx.DiGraph()

# 主网络节点索引范围
main_nodes = range(N_main)

# 随机创建多个调节网络（每个调节网络的大小在5到10之间随机）
subnet_nodes = {}
start_index = N_main  # 从主网络节点的最后一个节点之后开始

for i in range(N_subnets):
    subnet_size = 15 #np.random.randint(5,6) #调节网络的大小（5-10之间随机）
    subnet_nodes[i] = range(start_index, start_index + subnet_size)
    start_index += subnet_size

# 向每个调节网络添加节点
for nodes in subnet_nodes.values():
    G.add_nodes_from(nodes)

# 连接每个调节网络内部节点（每个网络内部有k条边）
for subnet in subnet_nodes.values():
    for node in subnet:
        targets = np.random.choice([n for n in subnet if n!= node], k, replace=False)
        for target in targets:
            G.add_edge(node, target)

# 连接主网络内部（随机连接，确保每个节点有k个边）
for node in main_nodes:
    targets = np.random.choice([n for n in main_nodes if n!= node], k, replace=False)
    for target in targets:
            G.add_edge(node, target)
            
# 设置随机输入，初始仅对一个节点施加强刺激
I = np.zeros(len(G.nodes))

# 创建传播权重矩阵（固定随机权重）
W = np.zeros((len(G.nodes), len(G.nodes)) )
for i, j in G.edges():
    W[i, j] = np.random.uniform(-1, 0)  # 权重在 0.8 到 1.2 之间波动
for i in range(0,N_main-1):
    for j in range(0,N_main-1):
        W[i,j] = np.random.uniform(-0.5,0.7)
for i in range(N_main,N_main+14):
    for j in range(N_main,N_main+14):
        W[i,j] = np.random.uniform(0.8,1.0)

# 动态绘制的设置

def add_edges_from_subnet_to_main(G, subnet_nodes, main_nodes):
    for subnet in subnet_nodes.values():
        num_edges = 3#np.random.randint(1, 2)
        subnet_nodes_list = list(subnet)
        main_nodes_list = list(main_nodes)
        subnet_node_choices = np.random.choice(subnet_nodes_list, num_edges, replace=False)
        main_node_choices = np.random.choice(main_nodes_list, num_edges, replace=False)
        for subnet_node, main_node in zip(subnet_node_choices, main_node_choices):
            if not G.has_edge(subnet_node, main_node):
                G.add_edge(subnet_node, main_node)
                W[subnet_node,main_node]=3


# 主网络向调节网络发出 1-3 条线
def add_edges_from_main_to_subnet(G, subnet_nodes, main_nodes):
    a_rand = np.random.randint(0, N_main-4)
    b_rand = 1 #np.random.randint(1, 2)
    main_nodes_slect = main_nodes[a_rand:a_rand+b_rand]
    for main_node in main_nodes_slect:
        num_edges = 1
        subnet_keys = list(subnet_nodes.keys())
        for _ in range(num_edges):
            target_subnet_index = np.random.choice(subnet_keys)
            target_subnet = subnet_nodes[target_subnet_index]
            target_subnet_node = np.random.choice(list(target_subnet))
            if not G.has_edge(main_node, target_subnet_node):
                G.add_edge(main_node, target_subnet_node)
                W[main_node,target_subnet_node]=3
                

'''# 调节网络向两个调节网络发出 1条边
def add_edges_between_subnets(G, subnet_nodes):
    subnet_keys = list(subnet_nodes.keys())
    for i, subnet in subnet_nodes.items():
        target_subnet_indices = np.random.choice([idx for idx in subnet_keys if idx!= i], 2, replace=False)
        num_edges = 1#np.random.randint(1, 3)
        subnet_nodes_list = list(subnet)
        subnet_node_choices = np.random.choice(subnet_nodes_list, num_edges, replace=False)
        for target_subnet_index in target_subnet_indices:
            target_subnet = subnet_nodes[target_subnet_index]
            target_subnet_nodes_list = list(target_subnet)
            target_subnet_node_choices = np.random.choice(target_subnet_nodes_list, num_edges, replace=False)
            for subnet_node, target_subnet_node in zip(subnet_node_choices, target_subnet_node_choices):
                if not G.has_edge(subnet_node, target_subnet_node):
                    G.add_edge(subnet_node, target_subnet_node)'''
                    


add_edges_from_subnet_to_main(G, subnet_nodes, main_nodes)
#add_edges_from_main_to_subnet(G, subnet_nodes, main_nodes)
#add_edges_between_subnets(G, subnet_nodes)

# 为每个节点分配三维坐标
layout = nx.spring_layout(G, dim=3, seed=42)  # 使用spring layout生成3D坐标
pos = {i: layout[i] for i in G.nodes}

# 初始化膜电位和恢复变量
V = np.full(len(G.nodes), -65.0, dtype=np.float64)  # 初始膜电位为-65 mV
V[-1] = 60
u = np.full(len(G.nodes), 0.0, dtype=np.float64)    # 初始恢复变量为0

# 随机化每个神经元的模型参数
params = {
    "a": np.random.uniform(0.02, 0.03, len(G.nodes)),
    "b": np.random.uniform(0.1, 0.3, len(G.nodes)),
    "c": np.random.uniform(-58, -56, len(G.nodes)),
    "d": np.random.uniform(2, 10, len(G.nodes)),
}


# 提取节点的三维坐标
xs, ys, zs = zip(*[pos[i] for i in G.nodes])

# 自定义颜色映射
colors_under_70 = plt.cm.inferno(np.linspace(0, 0.5, 128))  # -80到-70灰色渐变
colors_70_to_60 = plt.cm.plasma(np.linspace(0.4, 0.9, 128)) #-70到-50
colors_60_to_0 = plt.cm.cividis(np.linspace(0.6, 1.0, 128)) #-50到0冷色渐变
colors_0_to_80 = plt.cm.inferno(np.linspace(0.9, 1.0, 256) ) # 50到80暖色渐变
all_colors = np.vstack((colors_under_70, colors_70_to_60, colors_60_to_0, colors_0_to_80))
custom_cmap = ListedColormap(all_colors)

# 设置颜色的边界和分段
boundaries = [-80,-70,-65,-60,-55,-50,-40,-20,0,10,20,30,45,80]  # 边界递增
norm = BoundaryNorm(boundaries, custom_cmap.N, clip=True)

# 创建图形和3D子图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(121, projection='3d')
ax_phase = fig.add_subplot(122)  # 新增一个子图用于相图


# 初始化节点和边的绘图元素
sc = ax.scatter(xs, ys, zs, c=V, s=50, cmap=custom_cmap, norm=norm)

# 绘制有向边
lines = []
for i, j in G.edges():
    x_vals = [pos[i][0], pos[j][0]]
    y_vals = [pos[i][1], pos[j][1]]
    z_vals = [pos[i][2], pos[j][2]]
    line, = ax.plot(x_vals, y_vals, z_vals, color='gray', alpha=0.5, linestyle='-', linewidth=0.8)
    lines.append(line)


# 新增：存储两个随机选择的调节网络的平均电位
subnet_keys = list(subnet_nodes.keys())
#selected_subnets = np.random.choice(subnet_keys, 2, replace=False)

avg_potentials_x = []
avg_potentials_y = []
avg_potentials_z = []
avg_potentials_u = []


# 新增：初始化相图的散点图
phase_sc = ax_phase.scatter([], [], c='green', s=10)  # 设置散点大小为 10


# 添加颜色条
cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=custom_cmap), ax=ax, shrink=0.6)
cbar.set_label("Membrane Potential (mV)")
cbar.set_ticks(boundaries)  # 非线性分布刻度


# 更新函数
def update(t):
    global V, u, I, avg_potentials_x, avg_potentials_y

    # 动态更新每个节点的输入电流
    new_I = np.zeros(len(G.nodes))
    for i in range(len(G.nodes)):
        if V[i] > 0:  # 仅在膜电位大于0时向邻居传播刺激
            neighbors = list(G.successors(i))  # 获取出向邻居
            for j in neighbors:
                new_I[j] += W[i, j] * (V[i]*2 - params["c"][i])  # 根据膜电位生成刺激

    # 更新输入电流并添加随机噪声
    I = new_I #+ np.random.uniform(, len(G.nodes))

    # 更新膜电位和恢复变量
    for i in range(len(G.nodes)):
        V[i], u[i] = izhikevich(V[i], u[i], I[i], params["a"][i], params["b"][i], params["c"][i], params["d"][i])
        V[i] = np.clip(V[i], -80, 80)  # 限制膜电位的范围

    # 更新节点颜色
    sc.set_array(V)

    edgecolors = []
    for node in G.nodes():
        if node < N_main:
            edgecolors.append('blue')  # 主网络节点边框颜色为蓝色
        else:
            edgecolors.append('red')  # 调节网络节点边框颜色为红色
    sc.set_edgecolors(edgecolors)
    ax.set_title(f"Time Step: {t}")

    # 计算两个选定的调节网络的平均电位
    avg_potentials = []
    
    choosen_1 = [V[i] for i in range(N_main,N_main+5)]
    choosen_2 = [V[i] for i in range(N_main+5,N_main+10)]
    main_potentials = [V[i] for i in range(0,N_main)]
    reg_potentials = [V[i] for i in range(N_main,N_main+15)]
    avg_potentials_x.append(np.mean(choosen_1))
    avg_potentials_y.append(np.mean(choosen_2))
    avg_potentials_z.append(np.mean(main_potentials))
    avg_potentials_u.append(np.mean(reg_potentials))
    phase_sc.set_offsets(np.c_[avg_potentials_u, avg_potentials_z])  # 更新相图的散点图

    # 连接上一个点和当前点
    if len(avg_potentials_z) > 1:
        ax_phase.plot([avg_potentials_u[-2], avg_potentials_u[-1]], [avg_potentials_z[-2], avg_potentials_z[-1]], color='green')


    # 设置相图的坐标轴范围
    ax_phase.set_xlim(-80, 0)
    ax_phase.set_ylim(-80, 0)


    return sc, *lines


# 创建动画
ani = FuncAnimation(fig, update, frames=T, interval=200, blit=False)  # interval设置为200ms，放慢演化速度


# 保存动画为GIF文件
ani.save("izhikevich_multi_subnet_network.gif", writer="pillow")


# 显示动画
plt.show()

def plot_multiple_arrays(a, b, c, d):
    """
    此函数用于在同一幅图上绘制多个数组与 a 数组的关系

    参数:
    a (list or numpy.ndarray): 第一个数组
    b (list or numpy.ndarray): 第二个数组
    c (list or numpy.ndarray): 第三个数组
    d (list or numpy.ndarray): 第四个数组
    """
    # 检查输入是否为列表或 numpy 数组
    if not isinstance(a, (list, np.ndarray)) or not isinstance(b, (list, np.ndarray)) or not isinstance(c, (list, np.ndarray)) or not isinstance(d, (list, np.ndarray)):
        raise TypeError("输入必须是列表或 numpy 数组")
    # 检查所有数组的长度是否相等
    if len(a)!= len(b) or len(a)!= len(c) or len(a)!= len(d):
        raise ValueError("所有数组的长度必须相等"+str(len(a))+' '+str(len(b))+' '+str(len(c))+' '+str(len(d)))

    # 绘制图形
    plt.plot(a, b, label='b vs a')
    plt.plot(a, c, label='c vs a')
    plt.plot(a, d, label='d vs a')
    plt.xlabel('a 数组')
    plt.ylabel('其他数组')
    plt.title('多个数组与 a 数组的关系')
    plt.legend()  # 显示图例
    plt.grid(True)
    plt.show()

num=len(avg_potentials_z)
rang_a = np.array(range(num))/100
plot_multiple_arrays(rang_a, np.array(avg_potentials_z), np.array(avg_potentials_u), np.array(avg_potentials_x))
print(avg_potentials_z)

def plot_fourier_transform(A, B, C, D):
    """
    此函数将数组 B、C、D 相对于数组 A 进行傅里叶变换，
    并将结果绘制在同一图形中，重点关注频率范围在 [-0.02, 0.02] 内的部分。

    参数:
    A (list or numpy.ndarray): 基准数组
    B (list or numpy.ndarray): 第一个数组
    C (list or numpy.ndarray): 第二个数组
    D (list or numpy.ndarray): 第三个数组
    """
    # 将输入转换为 numpy 数组
    A = np.array(A)
    B = np.array(B)
    C = np.array(C)
    D = np.array(D)

    # 检查所有数组的长度是否相等
    if len(A)!= len(B) or len(A)!= len(C) or len(A)!= len(D):
        raise ValueError("所有数组的长度必须相等")

    # 进行傅里叶变换
    fft_B = np.fft.fft(B)
    fft_C = np.fft.fft(C)
    fft_D = np.fft.fft(D)

    # 计算频率
    freq = np.fft.fftfreq(len(A))

    # 找到频率范围在 [-0.02, 0.02] 内的索引
    mask = (freq >= -0.02) & (freq <= 0.02)
    freq_masked = freq[mask]
    fft_B_masked = fft_B[mask]
    fft_C_masked = fft_C[mask]
    fft_D_masked = fft_D[mask]

    # 绘制傅里叶变换结果
    plt.figure(figsize=(12, 6))
    plt.plot(freq_masked, np.abs(fft_B_masked), label='FFT of B')
    plt.plot(freq_masked, np.abs(fft_C_masked), label='FFT of C')
    plt.plot(freq_masked, np.abs(fft_D_masked), label='FFT of D')
    plt.title('Fourier Transform of B, C, and D in [-0.02, 0.02]')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 找到幅度最大的频率分量及其对应的频率（周期较大的规律）
    def find_max_peak(fft, freq):
        """
        此函数找到傅里叶变换结果中幅度最大的频率分量及其对应的频率。

        参数:
        fft (numpy.ndarray): 傅里叶变换结果
        freq (numpy.ndarray): 频率数组
        """
        index = np.argmax(np.abs(fft))
        max_peak = np.abs(fft[index])
        max_freq = freq[index]
        return max_peak, max_freq

    max_peak_B, max_freq_B = find_max_peak(fft_B_masked, freq_masked)
    max_peak_C, max_freq_C = find_max_peak(fft_C_masked, freq_masked)
    max_peak_D, max_freq_D = find_max_peak(fft_D_masked, freq_masked)

    print(f"For B: Max Peak Magnitude = {max_peak_B}, Max Peak Frequency = {max_freq_B}")
    print(f"For C: Max Peak Magnitude = {max_peak_C}, Max Peak Frequency = {max_freq_C}")
    print(f"For D: Max Peak Magnitude = {max_peak_D}, Max Peak Frequency = {max_freq_D}")

# 示例数据，可以是列表或 numpy 数组，也可以是其他可迭代对象
plot_fourier_transform(rang_a, np.array(avg_potentials_z), np.array(avg_potentials_u), np.array(avg_potentials_x))
