"""
Handle Posterior Probabilities using a Knob
===========================================
Interactive demos of Bayesian inference, Murphy (2022) Fig 3.9 style.

Tab 1 — 2D Gaussian (PRML Fig 3.9):
    y_n ~ N(z, Sigma_y), z_true = [0.5, 0.5]^T, prior N(0, 0.1 I).
    Knob = sensor noise Sigma_y = sigma^2 I.

Tab 2 — 1D Gaussian (two-stage knob):
    Data from N(mu=3, sigma=2), prior N(mu0=1, sigma0=1).
    Stage 1: turn the prior-mean knob to slide the posterior toward
             the rough location of the data.
    Stage 2: turn the prior-sigma knob to fine-tune the envelope /
             how tightly the posterior hugs the data.

The "knob" idea (Doing Bayesian Data Analysis): posterior probabilities
are a knob you turn to adjust the model's sensitivity to the data.

Run:  streamlit run app.py
"""

import time

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="Posterior Knob", layout="wide")

# ----------------------------------------------------------------------
# Shared sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("🎛️ Posterior Knob")
    st.markdown(
        "Turn the knobs to adjust how strongly the posterior "
        "reacts to the data (Murphy 2022, Fig 3.9; Kruschke's "
        "*DBDA* knob analogy)."
    )
    seed = st.number_input("Random seed", 0, 999, 42)
    st.divider()
    st.caption("Each tab has its own knobs. Data is regenerated only when the seed or N changes.")

# ======================================================================
# Shared math helpers
# ======================================================================
def gaussian_pdf_1d(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

# ======================================================================
# Tab 1 — 2D Gaussian (PRML Fig 3.9)
# ======================================================================
Z_TRUE = np.array([0.5, 0.5])
PRIOR_MEAN2D = np.zeros(2)
PRIOR_COV2D = 0.1 * np.eye(2)

GRID = np.linspace(-1.0, 1.0, 120)
XX, YY = np.meshgrid(GRID, GRID)
PTS = np.dstack([XX, YY])

def gaussian_pdf_2d(mean, cov):
    d = PTS - mean
    inv = np.linalg.inv(cov)
    maha = np.einsum("...i,ij,...j->...", d, inv, d)
    norm = 1.0 / (2 * np.pi * np.sqrt(np.linalg.det(cov)))
    return norm * np.exp(-0.5 * maha)

def contour_traces(mean, cov, color):
    """Nested contour ellipses (evenly spaced density levels)."""
    pdf = gaussian_pdf_2d(mean, cov)
    levels = np.linspace(pdf.max() * 0.05, pdf.max() * 0.95, 6)
    return go.Contour(
        x=GRID, y=GRID, z=pdf,
        showscale=False,
        contours=dict(start=levels[0], end=levels[-1],
                      size=levels[1] - levels[0], coloring="lines"),
        line=dict(color=color, width=1.5),
        colorscale=[[0, color], [1, color]],
        showlegend=False, hoverinfo="skip",
    )

def build_fig_2d(data, sigma_noise):
    """(a) data | (b) prior | (c) posterior — exact conjugate update."""
    n = len(data)
    Sigma_y = (sigma_noise ** 2) * np.eye(2)
    if n > 0:
        prior_prec = np.linalg.inv(PRIOR_COV2D)
        lik_prec = np.linalg.inv(Sigma_y)
        post_cov = np.linalg.inv(prior_prec + n * lik_prec)
        post_mean = post_cov @ (prior_prec @ PRIOR_MEAN2D + lik_prec @ data.sum(axis=0))
    else:
        post_cov, post_mean = PRIOR_COV2D.copy(), PRIOR_MEAN2D.copy()

    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.06)

    if n > 0:
        fig.add_trace(go.Scatter(
            x=data[:, 0], y=data[:, 1], mode="markers",
            marker=dict(color="royalblue", size=7),
            hovertemplate="(%{x:.2f}, %{y:.2f})<extra></extra>",
        ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[Z_TRUE[0]], y=[Z_TRUE[1]], mode="markers",
        marker=dict(symbol="x", color="black", size=14, line=dict(width=2)),
    ), row=1, col=1)
    fig.add_trace(contour_traces(PRIOR_MEAN2D, PRIOR_COV2D, "mediumseagreen"), row=1, col=2)
    fig.add_trace(contour_traces(post_mean, post_cov, "crimson"), row=1, col=3)
    fig.add_trace(go.Scatter(
        x=[post_mean[0]], y=[post_mean[1]], mode="markers",
        marker=dict(symbol="x", color="crimson", size=10),
    ), row=1, col=3)

    for i in range(1, 4):
        xa = f"xaxis{i if i > 1 else ''}"
        ya = f"yaxis{i if i > 1 else ''}"
        fig.layout[xa].update(range=[-1, 1], dtick=0.25, constrain="domain")
        fig.layout[ya].update(range=[-1, 1], dtick=0.25)
    fig.update_layout(
        height=380, showlegend=False, plot_bgcolor="white",
        margin=dict(l=10, r=10, t=40, b=10),
        annotations=[
            dict(text="(a) data", showarrow=False, font=dict(size=13), x=0.15, y=1.08, xref="paper", yref="paper"),
            dict(text="(b) prior", showarrow=False, font=dict(size=13), x=0.5, y=1.08, xref="paper", yref="paper"),
            dict(text=f"(c) posterior, N={n}", showarrow=False, font=dict(size=13), x=0.85, y=1.08, xref="paper", yref="paper"),
        ],
    )
    return fig, post_mean, post_cov

def draw_tab_2d():
    c1, c2 = st.columns(2)
    sigma_noise = c1.slider(
        "Knob: sensor noise σ (Σ_y = σ² I)", 0.05, 1.5, 0.10, 0.05,
        help="Small σ → posterior trusts the data. Large σ → posterior stays near the prior.",
    )
    n_obs = c2.slider("Number of observations N", 0, 50, 10)

    # data depends only on seed & N -> cache it
    @st.cache_data
    def gen_data_2d(seed, n):
        rng = np.random.default_rng(seed)
        return rng.multivariate_normal(Z_TRUE, (0.1 ** 2) * np.eye(2), size=n)

    data = gen_data_2d(int(seed), n_obs)
    fig, post_mean, _ = build_fig_2d(data, sigma_noise)
    st.plotly_chart(fig, width="stretch", key="fig2d")

    m1, m2, m3 = st.columns(3)
    m1.metric("knob: σ noise", f"{sigma_noise:.2f}")
    m2.metric("posterior mean", f"({post_mean[0]:.2f}, {post_mean[1]:.2f})")
    m3.metric("distance to truth", f"{np.linalg.norm(post_mean - Z_TRUE):.3f}")

    st.markdown(
        r"""
        **How the knob works** — exact conjugate update:

        $$\Sigma_{post} = (\Sigma_{prior}^{-1} + N\,\Sigma_y^{-1})^{-1}, \quad
          \mu_{post} = \Sigma_{post}\,(\Sigma_y^{-1} \textstyle\sum_n y_n)$$

        Turning **σ** down tightens $\Sigma_y^{-1}$: the posterior snaps toward the
        data (high sensitivity). Turning **σ** up flattens the likelihood and the
        posterior collapses onto the prior (low sensitivity).
        """
    )

# ======================================================================
# Tab 2 — 1D Gaussian, two-stage knob (mu first, then sigma)
# ======================================================================
MU_TRUE, SIGMA_TRUE = 3.0, 2.0   # data generator
MU0_DEFAULT, SIGMA0_DEFAULT = 1.0, 1.0  # prior N(1, 1)

def build_fig_1d(data, mu0, sigma0):
    """Histogram + prior + posterior; exact conjugate Gaussian update."""
    n = len(data)
    sigma2_like = SIGMA_TRUE ** 2
    if n > 0 and sigma0 > 0:
        post_var = 1.0 / (1.0 / sigma0 ** 2 + n / sigma2_like)
        post_mu = post_var * (mu0 / sigma0 ** 2 + data.sum() / sigma2_like)
        post_sd = np.sqrt(post_var)
    else:
        post_mu, post_sd = mu0, sigma0

    x = np.linspace(-6, 11, 600)
    fig = go.Figure()
    if n > 0:
        fig.add_trace(go.Histogram(
            x=data, histnorm="probability density", nbinsx=30,
            marker=dict(color="rgba(65,105,225,0.35)"),
            name="data", showlegend=False,
        ))
    fig.add_trace(go.Scatter(
        x=x, y=gaussian_pdf_1d(x, mu0, sigma0), mode="lines",
        line=dict(color="mediumseagreen", width=2, dash="dot"),
        name=f"prior N({mu0:.2f}, {sigma0:.2f}²)",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=gaussian_pdf_1d(x, post_mu, post_sd), mode="lines",
        line=dict(color="crimson", width=3),
        name="posterior",
    ))
    fig.add_vline(x=MU_TRUE, line=dict(color="black", width=1, dash="dash"),
                  annotation_text="true μ=3", annotation_position="top")
    fig.update_layout(
        height=420, showlegend=True, plot_bgcolor="white",
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(range=[-6, 11], title="x"),
        yaxis=dict(title="density"),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
    )
    return fig, post_mu, post_sd

def draw_tab_1d():
    st.markdown(
        "**Two-stage knob** — data come from $\\mathcal{N}(\\mu{=}3,\\ \\sigma{=}2)$, "
        "prior is $\\mathcal{N}(\\mu_0{=}1,\\ \\sigma_0{=}1)$. "
        "**Step 1:** slide μ₀ to find where the data roughly live. "
        "**Step 2:** slide σ₀ to fine-tune the envelope — a wide prior lets the "
        "data dominate, a narrow prior pins the posterior near μ₀."
    )
    c1, c2 = st.columns(2)
    mu0 = c1.slider("Stage 1 · prior mean μ₀", -3.0, 7.0, MU0_DEFAULT, 0.1,
                    help="Coarse localization: the posterior is pulled toward μ₀.")
    sigma0 = c2.slider("Stage 2 · prior σ₀", 0.2, 4.0, SIGMA0_DEFAULT, 0.05,
                       help="Envelope tuning: wide σ₀ → data dominate; narrow σ₀ → prior dominates.")

    @st.cache_data
    def gen_data_1d(seed, n):
        rng = np.random.default_rng(seed + 1)
        return rng.normal(MU_TRUE, SIGMA_TRUE, size=n)

    n_obs = st.slider("Number of observations N", 0, 200, 30)
    data = gen_data_1d(int(seed), n_obs)
    fig, post_mu, post_sd = build_fig_1d(data, mu0, sigma0)
    st.plotly_chart(fig, width="stretch", key="fig1d")

    data_mean = data.mean() if n_obs > 0 else float("nan")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("data mean", f"{data_mean:.2f}")
    m2.metric("prior", f"N({mu0:.2f}, {sigma0:.2f}²)")
    m3.metric("posterior", f"N({post_mu:.2f}, {post_sd:.2f}²)")
    m4.metric("posterior − data mean", f"{post_mu - data_mean:+.2f}")

    st.caption(
        "Exact conjugate update: "
        "σ²_post = (1/σ₀² + N/σ²)⁻¹,  μ_post = σ²_post (μ₀/σ₀² + Σx/σ²)."
    )

    # --- Auto demo: sweep mu0 first, then sigma0 (the two-stage routine)
    if st.toggle("▶ Auto demo: μ₀ first, then σ₀"):
        holder = st.empty()
        for m in np.linspace(MU0_DEFAULT, 3.0, 15):        # stage 1: find the data
            f, _, _ = build_fig_1d(data, m, SIGMA0_DEFAULT)
            f.update_layout(title=f"stage 1 — sliding μ₀ = {m:.2f}")
            holder.plotly_chart(f, width="stretch", key=f"mu_{m:.3f}")
            time.sleep(0.1)
        for s in np.linspace(SIGMA0_DEFAULT, 0.4, 15):     # stage 2: tighten envelope
            f, _, _ = build_fig_1d(data, 3.0, s)
            f.update_layout(title=f"stage 2 — tightening σ₀ = {s:.2f}")
            holder.plotly_chart(f, width="stretch", key=f"sg_{s:.3f}")
            time.sleep(0.1)

# ======================================================================
# Layout: tab-like switcher (segmented control keeps hidden views from
# recomputing — plain st.tabs always computes collapsed content)
# ======================================================================
view = st.segmented_control(
    "Demo", ["2D Gaussian · PRML Fig 3.9", "1D Gaussian · two-stage knob"],
    default="2D Gaussian · PRML Fig 3.9", label_visibility="collapsed",
)

if view == "2D Gaussian · PRML Fig 3.9":
    draw_tab_2d()
else:
    draw_tab_1d()
