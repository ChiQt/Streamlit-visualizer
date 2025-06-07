import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy import signal

# 设置页面布局
st.set_page_config(
    page_title="常见信号处理算法可视化",  # 算法参数可视化
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 标题和介绍
st.title("📈 信号处理算法（滤波器设计）可视化")     # 交互式算法参数可视化
st.markdown("""
调整右侧边栏的参数，实时查看算法输出结果的变化。
""")

# 创建带有标签的侧边栏参数控制
with st.sidebar:
    st.header("⚙️ 算法参数控制")
    
    # 算法类型选择
    algorithm_type = st.selectbox(
        "选择算法类型",
        ["低通滤波器", "带通滤波器", "高通滤波器"],
        index=0
    )
    
    # 公共参数
    sample_rate = st.slider("采样率 (Hz)", 100, 10000, 1000, 100)
    duration = st.slider("信号时长 (秒)", 0.1, 5.0, 1.0, 0.1)
    noise_level = st.slider("噪声水平", 0.0, 1.0, 0.2, 0.01)
    
    # 滤波器特定参数
    cutoff_freq = st.slider("截止频率 (Hz)", 1, 500, 50, 1)
    
    if "带通" in algorithm_type:
        bandwidth = st.slider("带宽 (Hz)", 10, 200, 50, 5)
    
    # 滤波器阶数
    filter_order = st.slider("滤波器阶数", 1, 10, 4, 1)
    
    # 信号类型选择
    signal_type = st.selectbox(
        "输入信号类型",
        ["正弦波", "方波", "锯齿波", "混合信号"],
        index=0
    )
    
    # 添加一些说明
    st.markdown("---")
    st.caption("调整参数后，可视化结果将自动更新")

# 生成原始信号
def generate_signal(signal_type, duration, sample_rate, noise_level):
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    if signal_type == "正弦波":
        sig = np.sin(2 * np.pi * 5 * t)  # 5Hz基础频率
    elif signal_type == "方波":
        sig = signal.square(2 * np.pi * 3 * t)
    elif signal_type == "锯齿波":
        sig = signal.sawtooth(2 * np.pi * 2 * t)
    else:  # 混合信号
        sig = (0.5 * np.sin(2 * np.pi * 2 * t) + 
               0.8 * np.sin(2 * np.pi * 8 * t) +
               0.3 * np.sin(2 * np.pi * 15 * t))
    
    # 添加噪声
    noise = noise_level * np.random.normal(size=len(t))
    return t, sig + noise

# 设计滤波器
def design_filter(algorithm_type, cutoff_freq, sample_rate, filter_order, bandwidth=None):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist

    if algorithm_type == "低通滤波器":
        b, a = signal.butter(filter_order, normal_cutoff, btype='low')
    elif algorithm_type == "高通滤波器":
        b, a = signal.butter(filter_order, normal_cutoff, btype='high')
    else:  # 带通滤波器
        low = (cutoff_freq - bandwidth/2) / nyquist
        high = (cutoff_freq + bandwidth/2) / nyquist
        b, a = signal.butter(filter_order, [low, high], btype='band')
    return b, a

# ✅ 注意这里参数是 input_signal 而不是 signal
def apply_filter(input_signal, b, a):
    return signal.filtfilt(b, a, input_signal)  # 使用 scipy.signal.filtfilt

# 计算频率响应
def frequency_response(b, a, sample_rate):
    w, h = signal.freqz(b, a, worN=2000)
    freqs = w * sample_rate / (2 * np.pi)
    return freqs, 20 * np.log10(np.abs(h))

# 生成数据
t, original_signal = generate_signal(signal_type, duration, sample_rate, noise_level)
b, a = design_filter(algorithm_type, cutoff_freq, sample_rate, filter_order, 
                     bandwidth if "带通" in algorithm_type else None)
filtered_signal = apply_filter(original_signal, b, a)
freqs, response = frequency_response(b, a, sample_rate)

# 创建多列布局
col1, col2 = st.columns(2)

# 在第一个列中显示时域信号
with col1:
    st.subheader("时域信号")
    
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(t, original_signal, label='Original signal', alpha=0.7)        
    ax1.plot(t, filtered_signal, label='The filtered signal', linewidth=2)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('amplitude')
    ax1.set_title('Before and after signal processing')  # 
    ax1.grid(True)
    ax1.legend()
    st.pyplot(fig1)
    
    # 添加一些统计信息
    st.write(f"**信号统计信息:**")
    stats_col1, stats_col2 = st.columns(2)
    with stats_col1:
        st.metric("原始信号标准差", f"{np.std(original_signal):.4f}")
    with stats_col2:
        st.metric("滤波后信号标准差", f"{np.std(filtered_signal):.4f}")

# 在第二个列中显示频域信息
with col2:
    st.subheader("频域分析")
    
    # 使用Plotly创建交互式频率响应图
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=freqs, y=response, mode='lines', name='频率响应'))
    fig2.update_layout(
        title='滤波器频率响应',   
        xaxis_title='频率 (Hz)',
        yaxis_title='增益 (dB)',
        height=400,
        hovermode="x"
    )
    # 添加截止频率标记
    if "带通" in algorithm_type:
        fig2.add_vline(x=cutoff_freq - bandwidth/2, line_dash="dash", line_color="red")
        fig2.add_vline(x=cutoff_freq + bandwidth/2, line_dash="dash", line_color="red")
        fig2.add_annotation(x=cutoff_freq, y=max(response), 
                            text=f"中心频率: {cutoff_freq}Hz", showarrow=True)
    else:
        fig2.add_vline(x=cutoff_freq, line_dash="dash", line_color="red", 
                       annotation_text=f"截止频率: {cutoff_freq}Hz")
    
    st.plotly_chart(fig2, use_container_width=True)
    
    # 频谱图
    st.subheader("信号频谱")
    
    fft_original = np.abs(np.fft.rfft(original_signal))
    fft_filtered = np.abs(np.fft.rfft(filtered_signal))
    freqs_fft = np.fft.rfftfreq(len(original_signal), 1/sample_rate)
    
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.semilogy(freqs_fft, fft_original, label='Original signal', alpha=0.7)
    ax3.semilogy(freqs_fft, fft_filtered, label='The filtered signal')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('amplitude (log)')
    ax3.set_title('Signal Spectrum')
    ax3.grid(True)
    ax3.legend()
    ax3.set_xlim(0, min(200, sample_rate/2))
    st.pyplot(fig3)

# 在底部添加滤波器系数
st.subheader("滤波器系数")
col_coeff1, col_coeff2 = st.columns(2)
with col_coeff1:
    st.write("**分子系数 (b):**")
    st.code(f"b = {b}")
with col_coeff2:
    st.write("**分母系数 (a):**")
    st.code(f"a = {a}")

# 添加一些说明
st.markdown("---")
st.info("""
**使用说明:**
1. 在左侧边栏调整算法参数
2. 可视化结果将实时更新
3. 可以切换不同的信号类型和滤波器类型
4. 图表支持交互操作（缩放、平移等）
""")

# 添加下载功能
if st.button("📥 导出当前配置为JSON"):
    config = {
        "algorithm_type": algorithm_type,
        "sample_rate": sample_rate,
        "duration": duration,
        "noise_level": noise_level,
        "cutoff_freq": cutoff_freq,
        "filter_order": filter_order,
        "signal_type": signal_type
    }
    if "带通" in algorithm_type:
        config["bandwidth"] = bandwidth
    
    st.download_button(
        label="下载配置",
        data=str(config),
        file_name="algorithm_config.json",
        mime="application/json"
    )