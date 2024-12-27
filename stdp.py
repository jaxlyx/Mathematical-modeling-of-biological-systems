import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import random

# Izhikevich模型参数
a = 0.02
b = 0.2
c = -65
d = 8
V_th = 30  # 阈值，单位mV
V_reset = -65  # 重置电位
U_reset = -14  # 重置恢复变量

# 神经元数量
n_neurons = 5

# 初始化神经元状态
V = np.full(n_neurons, -65.0)  # 每个神经元的初始膜电位
U = np.full(n_neurons, -14.0)  # 每个神经元的初始恢复变量
I = np.zeros(n_neurons)  # 外部输入电流

# 初始化神经元的发放时间（记录每个神经元的最后发放时间）
spike_times = np.full(n_neurons, -np.inf)  # 初始化为负无穷，表示没有发放

# 初始化神经元的三维位置 (随机分布在一个立方体中)
positions = np.random.rand(n_neurons, 3) * 10  # 位置范围：0-10 (单位：任意单位)

lestw=-2
mostw=2

# 初始化突触连接矩阵 (全连接有向图)
W = np.random.uniform(lestw, mostw, (n_neurons, n_neurons))  # 突触权重矩阵，-1到1之间
for k in range(n_neurons):
     for j in range(n_neurons):
         if j<k:
             W[k,j]=0#单向
         if j==k:
             W[j,k]=1
print(W)

# 时间参数
dt = 0.1  # 时间步长
T = 500  # 总模拟时间
time_steps = int(T / dt)
# STDP的参数
A_plus = 0.005  # 正向权重更新幅度
A_minus = 0.005  # 负向权重更新幅度
tau_plus = 20.0  # 正向STDP的时间常数（ms）
tau_minus = 20.0  # 负向STDP的时间常数（ms）

# 记录结果
V_record = np.zeros((n_neurons, time_steps))  # 每个神经元在每个时间点的膜电位
W_record = np.zeros((n_neurons, n_neurons, time_steps))  # 记录每个连接的突触权重变化
V_average = np.zeros(time_steps)

# 神经元的放电状态 (1为放电，0为不放电)
S = np.zeros(n_neurons)

# 创建膜电位变化图的子图
fig_time = plt.figure(figsize=(15, 8))
ax_time = fig_time.subplots(n_neurons, 1, sharex=True)

# 设置子图标题
for i in range(n_neurons):
    ax_time[i].set_title(f"Neuron {i+1} Membrane Potential")
    ax_time[i].set_ylabel('V (mV)')

ax_time[-1].set_xlabel('Time (ms)')

# 创建神经元分布和突触连接的3D图
fig_3d = plt.figure(figsize=(10, 8))
ax_3d = fig_3d.add_subplot(111, projection='3d')

# 初始颜色映射
colors = plt.cm.viridis((V - V_reset) / (V_th - V_reset))

# 绘制神经元的三维位置
scat = ax_3d.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=colors, s=200, marker='o')

# 绘制神经元之间的连接边
edge_lines = []
for i in range(n_neurons):
    for j in range(n_neurons):
        if W[i, j] != 0:  # 如果有连接
            # 使用权重控制边的颜色和线的粗细
            weight = W[i, j]
            color = 'r' if weight > 0 else 'b'  # 红色表示正向连接，蓝色表示负向连接
            linewidth =  abs(weight)  # 连接线的粗细，权重大时线条更粗

            # 绘制一条连接线
            line, = ax_3d.plot([positions[i, 0], positions[j, 0]], 
                               [positions[i, 1], positions[j, 1]], 
                               [positions[i, 2], positions[j, 2]], 
                               color=color, linewidth=linewidth)
            edge_lines.append(line)

# 设置3D图的标签
ax_3d.set_xlabel('X')
ax_3d.set_ylabel('Y')
ax_3d.set_zlabel('Z')
ax_3d.set_title('Neurons in 3D Space with Membrane Potential and Synaptic Connections')

# 添加颜色条
cbar = plt.colorbar(scat)
cbar.set_label('Membrane Potential (mV)')

# 添加当前时间的文本标签
time_text = ax_3d.text2D(0.95, 0.95, f'Time: {0:.2f} ms', transform=ax_3d.transAxes, ha='right', va='top', fontsize=18)

def update_synaptic_weights():
    global W, spike_times

    for i in range(n_neurons):
        for j in range(n_neurons):
            if i != j:
                # 如果任意一个神经元还没有发放（spike_times[i] 或 spike_times[j] 为 -np.inf），跳过
                if spike_times[i] == -np.inf or spike_times[j] == -np.inf:
                    continue

                # 计算发放时间差（单位ms）
                delta_t = spike_times[i] - spike_times[j]

                # 更新突触权重
                if delta_t > 0:
                    W[i, j] += A_plus * np.exp(-delta_t / tau_plus)
                elif delta_t < 0:
                    W[i, j] -= A_minus * np.exp(-delta_t / tau_minus)

                # 确保权重在[-5, 5]范围内
                W[i, j] = np.clip(W[i, j], lestw, mostw)


# 外部电流刺激函数
def get_external_current(t, i):
    """
    返回给定时间 t 和神经元 i 的外部电流。
    你可以根据需要修改这个函数，模拟不同的输入模式。
    """
    # 示例：简单的周期性刺激，每个神经元有不同的周期
    if  T/3<t<T*2/3 :
        I = 10+ random.gauss(1, 1) # 每个神经元的刺激频率不同
        amplitude = 5  # 刺激幅值
    else:
        I = random.gauss(1, 1)
    return 10+ random.gauss(1, 1)  # 时间单位是毫秒

# 计算突触电流（高斯分布）
def synaptic_current(S, W, i):
    """
    计算从神经元 i 到其他神经元的突触电流。
    这里的电流是一个高斯分布。
    """
    current = 0
    for j in range(n_neurons):
        if S[j] == 1:  # 只有在神经元j发放时才会传递电流
            # 使用突触权重生成一个高斯分布的电流，均值为0，标准差与权重相关
            current += np.random.normal(W[i, j], 0.01)  # 0.5 是标准差，可以根据需要调整
    return current

# 更新函数，用于动画
def update(frame):
    global V, U, S, I

    # 当前时间
    t = frame * dt

    # 更新神经元的状态
    for i in range(n_neurons):
        # 计算外部电流和来自其他神经元的影响
        I[i] = get_external_current(t, i) + synaptic_current(S, W, i)  # 高斯分布的突触电流

        # Izhikevich模型方程
        V[i] += 0.04 * V[i] ** 2 + 5 * V[i] + 140 - U[i] + I[i]
        U[i] += a * (b * V[i] - U[i])

        # 检查是否超出了阈值
        if V[i] >= V_th:
            V[i] = c  # 重置膜电位
            U[i] += d  # 重置恢复变量
            S[i] = 1  # 神经元放电
            spike_times[i]=t
        else:
            S[i] = 0  # 神经元不放电

        # 记录膜电位
        V_record[i, frame] = V[i]
        V_average[frame] =V_average[frame] +V[i]
    V_average[frame]=V_average[frame]/n_neurons
    # 更新膜电位颜色
    colors = plt.cm.viridis((V - V_reset) / (V_th - V_reset))
    scat.set_facecolor(colors)
    
    # 更新突触权重
    update_synaptic_weights()

    # 记录突触权重的变化
    W_record[:, :, frame] = W

    # 更新连接线颜色和粗细
    for i in range(n_neurons):
        for j in range(n_neurons):
            if W[i, j] != 0:  # 如果有连接
                weight = W[i, j]
                color = 'r' if weight > 0 else 'b'  # 红色表示正向连接，蓝色表示负向连接
                linewidth = 1 + abs(weight)  # 连接线的粗细，权重大时线条更粗
                edge_lines[i * n_neurons + j - (int)(i * (i + 1) / 2)].set_color(color)
                edge_lines[i * n_neurons + j - (int)(i * (i + 1) / 2)].set_linewidth(linewidth)

    # 更新文本显示当前时间
    time_text.set_text(f'Time: {frame * dt:.2f} ms')

    return scat, *edge_lines, time_text  # 返回更新后的元素


def plot_weight_changes(W_record, time_steps):
    plt.figure(figsize=(15, 10))

    for i in range(n_neurons):
        for j in range(n_neurons):
            if W[i, j] != 0 and i!=j:  # 只绘制有连接的权重变化
                plt.plot(np.arange(time_steps) * dt, W_record[i, j, :], label=f"Connection ({i+1} -> {j+1})")

    plt.xlabel('Time (ms)')
    plt.ylabel('Synaptic Weight')
    plt.title('Synaptic Weights over Time')
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1))
    plt.grid(True)
    plt.show()

# 绘制前三个神经元的三维膜电位相图
def plot_3d_phase_space(V_record, n_neurons, time_steps, dt):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 提取前三个神经元的膜电位
    V1 = V_record[0, :]
    V2 = V_record[1, :]
    V3 = V_record[2, :]

    # 绘制三维相图：三个神经元的膜电位随时间的变化
    ax.plot(V1, V2, V3, label='Phase Space of Neurons 1, 2, 3')

    # 设置标签
    ax.set_xlabel('Neuron 1 Membrane Potential V1 (mV)')
    ax.set_ylabel('Neuron 2 Membrane Potential V2 (mV)')
    ax.set_zlabel('Neuron 3 Membrane Potential V3 (mV)')
    ax.set_title('3D Phase Space of the First Three Neurons')
    ax.legend()
    plt.show()

# 假设你已经有了膜电位记录 V_record 变量（形状为 [n_neurons, time_steps]）

def compute_fourier_transform(V_record, time_steps, dt):
    """
    对每个神经元的膜电位信号进行傅里叶变换，得到频谱信息。
    V_record: 神经元膜电位记录，形状为 [n_neurons, time_steps]
    time_steps: 总时间步数
    dt: 时间步长
    """
    # 计算频率轴
    freqs = np.fft.fftfreq(time_steps, dt)  # 频率范围，从负频率到正频率
    freqs = np.fft.fftshift(freqs)  # 将频率移到正频率部分

    # 傅里叶变换结果
    V_fft = np.fft.fft(V_record, axis=1)  # 对每个神经元进行傅里叶变换（按时间轴方向）
    V_fft_shifted = np.fft.fftshift(V_fft, axes=1)  # 移动频谱到正频率部分

    # 计算幅度谱（绝对值）
    V_amp = np.abs(V_fft_shifted)

    return freqs, V_amp

def plot_frequency_spectrum(freqs, V_amp, n_neurons):
    """
    绘制每个神经元的频谱图。
    freqs: 频率轴
    V_amp: 幅度谱
    n_neurons: 神经元数量
    """
    plt.figure(figsize=(15, 15))
    
    for i in range(n_neurons):
        plt.subplot(n_neurons, 1, i+1)
        plt.plot(freqs, V_amp[i, :])
        #plt.title(f"Neuron {i+1} Frequency Spectrum")
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.xlim(0, 10)  # 频率范围可以根据需要调整
        plt.ylim(0,100)
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# 运行模拟并收集所有数据
for frame in range(time_steps):
    update(frame)

# 创建动画
ani = FuncAnimation(fig_3d, update, frames=range(time_steps), interval=dt, blit=True)


# 生成每个神经元的膜电位随时间变化的静态图
for i in range(n_neurons):
    ax_time[i].plot(np.arange(time_steps) * dt, V_record[i, :], label=f'Neuron {i+1}')

# 显示静态图
plt.tight_layout()


# 绘制前三个神经元的三维相图
plot_3d_phase_space(V_record, n_neurons, time_steps, dt)


# 计算傅里叶变换和幅度谱
freqs, V_amp = compute_fourier_transform(V_record, time_steps, dt)

# 绘制突触权重随时间的变化图
plot_weight_changes(W_record, time_steps)

# 绘制频谱图
plot_frequency_spectrum(freqs, V_amp, n_neurons)
plt.show()  

t=np.arange(0, T, dt)
plt.figure(figsize=(8, 1))
plt.plot(t, V_average, lw=1.5, color='red')  # 轨迹线加粗
plt.title('Membrane potential vs. Time')
plt.xlabel('Time [ms]')
plt.ylabel('Membrane potential (V) [mV]')
plt.grid(True)
plt.show()

