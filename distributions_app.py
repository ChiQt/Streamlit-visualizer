
import streamlit as st
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

st.set_page_config(page_title="Distribution Explorer", layout="wide")
st.title("📊 Distribution Explorer")
st.markdown(
    "Interactively explore probability distributions. "
    "Use the sidebar to pick a distribution and adjust its parameters."
)

# --- Sidebar controls ----------------------------------------------------
dist_name = st.sidebar.selectbox(
    "Select distribution",
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

# Helper to draw plot
def draw_pdf_cdf(x, pdf, cdf, discrete=False):
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        if discrete:
            ax.vlines(x, 0, pdf, lw=2)
            ax.plot(x, pdf, "o")
        else:
            ax.plot(x, pdf, lw=2)
        ax.set_title("Probability Density / Mass Function")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        if discrete:
            ax.step(x, cdf, where="post", lw=2)
        else:
            ax.plot(x, cdf, lw=2)
        ax.set_title("Cumulative Distribution Function")
        ax.set_xlabel("x")
        ax.set_ylabel("F(x)")
        st.pyplot(fig)

# --- Parameter widgets & plotting ----------------------------------------
if dist_name == "Normal":
    mu = st.sidebar.slider("μ (mean)", -10.0, 10.0, 0.0, 0.1)
    sigma = st.sidebar.slider("σ (std dev)", 0.1, 10.0, 1.0, 0.1)
    x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 400)
    pdf = stats.norm.pdf(x, mu, sigma)
    cdf = stats.norm.cdf(x, mu, sigma)
    draw_pdf_cdf(x, pdf, cdf)

elif dist_name == "Exponential":
    lam = st.sidebar.slider("λ (rate)", 0.1, 5.0, 1.0, 0.1)
    x = np.linspace(0, 10 / lam, 400)
    pdf = stats.expon.pdf(x, scale=1 / lam)
    cdf = stats.expon.cdf(x, scale=1 / lam)
    draw_pdf_cdf(x, pdf, cdf)

elif dist_name == "Poisson":
    mu = st.sidebar.slider("λ (mean)", 0.0, 20.0, 4.0, 0.5)
    k_max = int(mu + 4 * np.sqrt(mu + 1e-6)) + 5
    k = np.arange(0, k_max + 1)
    pmf = stats.poisson.pmf(k, mu)
    cdf = stats.poisson.cdf(k, mu)
    draw_pdf_cdf(k, pmf, cdf, discrete=True)

elif dist_name == "Binomial":
    n = st.sidebar.slider("n (trials)", 1, 100, 20, 1)
    p = st.sidebar.slider("p (success probability)", 0.0, 1.0, 0.5, 0.01)
    k = np.arange(0, n + 1)
    pmf = stats.binom.pmf(k, n, p)
    cdf = stats.binom.cdf(k, n, p)
    draw_pdf_cdf(k, pmf, cdf, discrete=True)

elif dist_name == "Uniform":
    a = st.sidebar.slider("a (lower)", -10.0, 9.0, 0.0, 0.5)
    b = st.sidebar.slider("b (upper)", a + 0.1, 10.0, a + 1.0, 0.5)
    x = np.linspace(a - 1, b + 1, 400)
    pdf = stats.uniform.pdf(x, loc=a, scale=b - a)
    cdf = stats.uniform.cdf(x, loc=a, scale=b - a)
    draw_pdf_cdf(x, pdf, cdf)

elif dist_name == "Beta":
    alpha = st.sidebar.slider("α (alpha)", 0.1, 10.0, 2.0, 0.1)
    beta = st.sidebar.slider("β (beta)", 0.1, 10.0, 2.0, 0.1)
    x = np.linspace(0, 1, 400)
    pdf = stats.beta.pdf(x, alpha, beta)
    cdf = stats.beta.cdf(x, alpha, beta)
    draw_pdf_cdf(x, pdf, cdf)

elif dist_name == "Gamma":
    k = st.sidebar.slider("k (shape)", 0.1, 10.0, 2.0, 0.1)
    theta = st.sidebar.slider("θ (scale)", 0.1, 10.0, 2.0, 0.1)
    x = np.linspace(0, stats.gamma.ppf(0.995, k, scale=theta), 400)
    pdf = stats.gamma.pdf(x, k, scale=theta)
    cdf = stats.gamma.cdf(x, k, scale=theta)
    draw_pdf_cdf(x, pdf, cdf)

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️")
