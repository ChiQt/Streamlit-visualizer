import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import plotly.graph_objects as go

###############################################################################
# 🚀 Distribution Explorer+                                                   #
# An upgraded interactive playground for probability distributions.          #
# Features:                                                                  #
#   • Clean two‑column layout with themed PDF/PMF, CDF charts                #
#   • Optional random‑sample generator & downloadable CSV                    #
#   • On‑demand summary statistics                                           #
#   • Customisable Matplotlib style palette                                  #
#   • Keyboard‑friendly sidebar (ȹ for parameters)                           #
###############################################################################

st.set_page_config(page_title="📊 Distribution Explorer+", page_icon="📈", layout="wide")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
col_logo, col_title = st.columns([0.1, 0.9])
with col_logo:
    st.markdown("## 📈")
with col_title:
    st.markdown("# Distribution Explorer+")
    st.markdown("Interactively explore probability distributions, inspect statistics, and generate random samples.")

st.markdown("---")

# ---------------------------------------------------------------------------
# Sidebar – global controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Controls")

    dist_name = st.selectbox(
        "Distribution",
        (
            "Normal",
            "Exponential",
            "Poisson",
            "Binomial",
            "Uniform",
            "Beta",
            "Gamma",
        ),
    )

    # Optional widgets
    show_stats = st.checkbox("Show summary statistics", value=True)
    show_sample = st.checkbox("Generate random sample", value=False)
    if show_sample:
        sample_size = st.number_input("Sample size", 10, 100_000, 1_000, step=10)

    # Matplotlib style selector
    style = st.selectbox("Plot style", options=["default", "ggplot", "seaborn-v0_8", "fivethirtyeight"])
    plt.style.use(style)

# Helper --------------------------------------------------------------------

def draw_pdf_cdf(x, pdf, cdf, discrete=False):
    """Render side‑by‑side PDF/PMF and CDF charts using Matplotlib."""
    col1, col2 = st.columns(2)

    # PDF / PMF
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        if discrete:
            ax.vlines(x, 0, pdf, lw=2)
            ax.plot(x, pdf, "o")
        else:
            ax.plot(x, pdf, lw=2)
        ax.set_title("Probability Density / Mass Function")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.grid(True, ls=":", lw=0.5)
        st.pyplot(fig)

    # CDF
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        if discrete:
            ax.step(x, cdf, where="post", lw=2)
        else:
            ax.plot(x, cdf, lw=2)
        ax.set_title("Cumulative Distribution Function")
        ax.set_xlabel("x")
        ax.set_ylabel("F(x)")
        ax.grid(True, ls=":", lw=0.5)
        st.pyplot(fig)


def summarise(sample):
    """Return key statistics as a DataFrame for display/download."""
    return pd.DataFrame({
        "Statistic": ["Mean", "Std Dev", "Median", "Skew", "Kurtosis"],
        "Value": [
            np.mean(sample),
            np.std(sample, ddof=1),
            np.median(sample),
            stats.skew(sample),
            stats.kurtosis(sample, fisher=False),
        ],
    })

# ---------------------------------------------------------------------------
# Distribution‑specific parameter widgets & calculations
# ---------------------------------------------------------------------------

def normal_section():
    mu = st.sidebar.slider("μ (mean)", -10.0, 10.0, 0.0, 0.1)
    sigma = st.sidebar.slider("σ (std dev)", 0.1, 10.0, 1.0, 0.1)
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 400)
    pdf = stats.norm.pdf(x, mu, sigma)
    cdf = stats.norm.cdf(x, mu, sigma)
    draw_pdf_cdf(x, pdf, cdf)
    sample = stats.norm.rvs(mu, sigma, size=sample_size) if show_sample else None
    return sample

def exponential_section():
    lam = st.sidebar.slider("λ (rate)", 0.1, 5.0, 1.0, 0.1)
    x = np.linspace(0, 10 / lam, 400)
    pdf = stats.expon.pdf(x, scale=1 / lam)
    cdf = stats.expon.cdf(x, scale=1 / lam)
    draw_pdf_cdf(x, pdf, cdf)
    sample = stats.expon.rvs(scale=1 / lam, size=sample_size) if show_sample else None
    return sample

def poisson_section():
    mu = st.sidebar.slider("λ (mean)", 0.0, 20.0, 4.0, 0.5)
    k_max = int(mu + 4 * np.sqrt(mu + 1e-6)) + 5
    k = np.arange(0, k_max + 1)
    pmf = stats.poisson.pmf(k, mu)
    cdf = stats.poisson.cdf(k, mu)
    draw_pdf_cdf(k, pmf, cdf, discrete=True)
    sample = stats.poisson.rvs(mu, size=sample_size) if show_sample else None
    return sample

def binomial_section():
    n = st.sidebar.slider("n (trials)", 1, 100, 20, 1)
    p = st.sidebar.slider("p (success prob)", 0.0, 1.0, 0.5, 0.01)
    k = np.arange(0, n + 1)
    pmf = stats.binom.pmf(k, n, p)
    cdf = stats.binom.cdf(k, n, p)
    draw_pdf_cdf(k, pmf, cdf, discrete=True)
    sample = stats.binom.rvs(n, p, size=sample_size) if show_sample else None
    return sample

def uniform_section():
    a = st.sidebar.slider("a (lower)", -10.0, 9.0, 0.0, 0.5)
    b = st.sidebar.slider("b (upper)", a + 0.1, 10.0, a + 1.0, 0.5)
    x = np.linspace(a - 1, b + 1, 400)
    pdf = stats.uniform.pdf(x, loc=a, scale=b - a)
    cdf = stats.uniform.cdf(x, loc=a, scale=b - a)
    draw_pdf_cdf(x, pdf, cdf)
    sample = stats.uniform.rvs(loc=a, scale=b - a, size=sample_size) if show_sample else None
    return sample

def beta_section():
    alpha = st.sidebar.slider("α (alpha)", 0.1, 10.0, 2.0, 0.1)
    beta = st.sidebar.slider("β (beta)", 0.1, 10.0, 2.0, 0.1)
    x = np.linspace(0, 1, 400)
    pdf = stats.beta.pdf(x, alpha, beta)
    cdf = stats.beta.cdf(x, alpha, beta)
    draw_pdf_cdf(x, pdf, cdf)
    sample = stats.beta.rvs(alpha, beta, size=sample_size) if show_sample else None
    return sample

def gamma_section():
    k_shape = st.sidebar.slider("k (shape)", 0.1, 10.0, 2.0, 0.1)
    theta = st.sidebar.slider("θ (scale)", 0.1, 10.0, 2.0, 0.1)
    x = np.linspace(0, stats.gamma.ppf(0.995, k_shape, scale=theta), 400)
    pdf = stats.gamma.pdf(x, k_shape, scale=theta)
    cdf = stats.gamma.cdf(x, k_shape, scale=theta)
    draw_pdf_cdf(x, pdf, cdf)
    sample = stats.gamma.rvs(k_shape, scale=theta, size=sample_size) if show_sample else None
    return sample

# Map distribution to handler
section_map = {
    "Normal": normal_section,
    "Exponential": exponential_section,
    "Poisson": poisson_section,
    "Binomial": binomial_section,
    "Uniform": uniform_section,
    "Beta": beta_section,
    "Gamma": gamma_section,
}

# Execute selected section ---------------------------------------------------

sample_data = section_map[dist_name]()

# ---------------------------------------------------------------------------
# Statistics & samples display
# ---------------------------------------------------------------------------

if show_sample and sample_data is not None:
    st.markdown("### 📋 Random Sample Histogram")
    fig = go.Figure(data=[go.Histogram(x=sample_data, nbinsx=50)])
    fig.update_layout(bargap=0.05, height=300, template="simple_white")
    st.plotly_chart(fig, use_container_width=True)

    # Download link
    csv = pd.DataFrame({"sample": sample_data}).to_csv(index=False).encode()
    st.download_button("Download sample as CSV", csv, file_name=f"{dist_name}_sample.csv")

if show_stats and sample_data is not None:
    st.markdown("### 🧮 Summary statistics")
    st.table(summarise(sample_data))

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption("Made with ❤️ by Distribution Explorer+")