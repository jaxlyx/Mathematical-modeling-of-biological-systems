import numpy as np
import matplotlib.pyplot as plt


def izhikevich_neuron(v, u, I, a, b, c, d, dt):
    """
    Izhikevich神经元模型的更新函数
    :param v: 膜电位
    :param u: 恢复变量
    :param I: 输入电流
    :param a: 模型参数
    :param b: 模型参数
    :param c: 重置膜电位
    :param d: 重置恢复变量
    :param dt: 时间步长
    :return: 更新后的v, u
    """
    dv = (0.04 * v ** 2 + 5 * v + 140 - u + I) * dt
    du = (a * (b * v - u)) * dt
    v_new = v + dv
    u_new = u + du
    if v_new >= 30:
        v_new = c
        u_new = u_new + d
    return v_new, u_new


def stdp_update(g_ij, t_i, t_j, A_plus, A_minus, tau_plus, tau_minus):
    """
    STDP规则更新耦合系数g_ij
    :param g_ij: 耦合系数
    :param t_i: 神经元i的发放时间
    :param t_j: 神经元j的发放时间
    :param A_plus: 正向学习率常数
    :param A_minus: 负向学习率常数
    :param tau_plus: 正向时间常数
    :param tau_minus: 负向时间常数
    :return: 更新后的g_ij
    """
    delta_t = t_j - t_i
    if delta_t > 0:
        delta_g = A_plus * np.exp(-delta_t / tau_plus)
    else:
        delta_g = -A_minus * np.exp(delta_t / tau_minus)
    g_ij += delta_g
    return g_ij


def kuramoto_parameter(theta):
    """
    计算Kuramoto参数
    :param theta: 相位数组
    :return: Kuramoto序参数r和平均相位psi
    """
    sum_cos = np.sum(np.cos(theta))
    sum_sin = np.sum(np.sin(theta))
    r = np.sqrt(sum_cos ** 2 + sum_sin ** 2) / len(theta)
    psi = np.arctan2(sum_sin, sum_cos)
    return r, psi


def main():
    # 模拟参数
    N = 100  # 神经元数量
    T = 1000  # 模拟时间
    dt = 0.1  # 时间步长
    a = 0.02
    b = 0.2
    c = -65
    d = 8
    A_plus = 0.1
    A_minus = 0.1
    tau_plus = 20
    tau_minus = 20
    g = np.random.rand(N, N) * 0.1  # 初始耦合系数矩阵
    v = np.random.rand(N) * (-65)  # 初始膜电位
    u = np.random.rand(N) * b * (-65)  # 初始恢复变量
    theta = np.random.rand(N) * 2 * np.pi  # 初始相位
    spike_times = [[] for _ in range(N)]  # 存储每个神经元的脉冲发放时间
    g_history = []  # 存储耦合系数的历史
    r_history = []  # 存储Kuramoto序参数的历史

    for t in np.arange(0, T, dt):
        I = np.zeros(N)
        for i in range(N):
            for j in range(N):
                if i!= j:
                    # 简单的脉冲发放信号，仅在发放时为1
                    s_j = 1 if len(spike_times[j]) > 0 and t == spike_times[j][-1] else 0
                    I[i] += g[i, j] * s_j
        for i in range(N):
            v[i], u[i] = izhikevich_neuron(v[i], u[i], I[i], a, b, c, d, dt)
            if v[i] >= 30:
                spike_times[i].append(t)
                # 基于STDP更新耦合系数
                for j in range(N):
                    if j!= i and len(spike_times[j]) > 0:
                        last_spike_j = spike_times[j][-1]
                        g[i, j] = stdp_update(g[i, j], t, last_spike_j, A_plus, A_minus, tau_plus, tau_minus)
        # 更新相位
        for i in range(N):
            if len(spike_times[i]) > 0 and t >= spike_times[i][-1]:
                theta[i] += 2 * np.pi * (t - spike_times[i][-1])
        r, psi = kuramoto_parameter(theta)
        g_history.append(g.copy())
        r_history.append(r)

    # 可视化结果
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(g, cmap='hot', interpolation='nearest')
    plt.title('Final Coupling Coefficient Matrix g')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(0, T, dt), r_history)
    plt.title('Kuramoto Order Parameter r over Time')
    plt.xlabel('Time')
    plt.ylabel('r')
    plt.show()


if __name__ == "__main__":
    main()
