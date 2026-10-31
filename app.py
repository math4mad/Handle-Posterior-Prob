"""
Handle Posterior Probabilities using a Knob
============================================
Interactive recreation of PRML / Murphy (2022) Figure 3.9:

    Bayesian inference for a 2d Gaussian random vector z.
      (a) data y_n ~ N(z, Sigma_y),  z_true = [0.5, 0.5]^T
      (b) prior    p(z) = N(0, 0.1 I)
      (c) posterior after N observations

The "knob" idea (Doing Bayesian Data Analysis): posterior probabilities are
a knob you turn to adjust the model's sensitivity to the data. Here the
knob is the sensor noise covariance Sigma_y = sigma^2 I -- turn it down
and the posterior hugs the data; turn it up and the prior dominates.

Run:  streamlit run app.py
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="Posterior Knob — PRML Fig 3.9", layout="wide")

# ----------------------------------------------------------------------
# Ground truth (matches figure 3.9 caption)
# ----------------------------------------------------------------------
Z_TRUE = np.array([0.5, 0.5])
PRIOR_MEAN = np.zeros(2)
PRIOR_COV = 0.1 * np.eye(2)

# ----------------------------------------------------------------------
# Knobs (the "posterior knob" panel)
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("🎛️ Posterior Knob")
    st.markdown(
        "Turn the knobs to adjust how strongly the posterior "
        "reacts to the data (Murphy 2022, Fig 3.9)."
    )

    sigma_noise = st.slider(
        "Sensor noise σ (Σ_y = σ² I)",
        0.05, 1.5, 0.10, 0.05,
        help="Small σ → posterior trusts the data. Large σ → posterior stays near the prior.",
    )
    n_obs = st.slider(
        "Number of observations N",
        0, 50, 10,
        help="How many data points have been seen (Fig 3.9c uses N=10).",
    )
    seed = st.number_input("Random seed", 0, 999, 42)

    st.divider()
    st.markdown("**Auto-play animation**")
    animate = st.toggle("Animate the knob", value=False)
    frames = st.slider("Animation frames", 5, 60, 25)

# ----------------------------------------------------------------------
# Data + exact conjugate posterior  (Gaussian-Gaussian)
# ----------------------------------------------------------------------
rng = np.random.default_rng(seed)
Sigma_y = (sigma_noise ** 2) * np.eye(2)
data = rng.multivariate_normal(Z_TRUE, Sigma_y, size=n_obs)  # (N,2)

# Posterior of z given y_1..y_N:
#   precision form: Σ_post = (Σ_prior^{-1} + N Σ_y^{-1})^{-1}
#   μ_post = Σ_post (Σ_prior^{-1} μ_prior + Σ_y^{-1} Σ y_n)
if n_obs > 0:
    prior_prec = np.linalg.inv(PRIOR_COV)
    lik_prec = np.linalg.inv(Sigma_y)
    post_cov = np.linalg.inv(prior_prec + n_obs * lik_prec)
    post_mean = post_cov @ (prior_prec @ PRIOR_MEAN + lik_prec @ data.sum(axis=0))
else:
    post_cov, post_mean = PRIOR_COV.copy(), PRIOR_MEAN.copy()

# ----------------------------------------------------------------------
# Contour helper -- draws ~6 shrinking ellipses like the book figure
# ----------------------------------------------------------------------
GRID = np.linspace(-1.0, 1.0, 120)
XX, YY = np.meshgrid(GRID, GRID)
PTS = np.dstack([XX, YY])

def gaussian_pdf(mean, cov):
    d = PTS - mean
    inv = np.linalg.inv(cov)
    maha = np.einsum("...i,ij,...j->...", d, inv, d)
    norm = 1.0 / (2 * np.pi * np.sqrt(np.linalg.det(cov)))
    return norm * np.exp(-0.5 * maha)

def contour_traces(mean, cov, color, label):
    """Nested contour ellipses (quartiles of the Gaussian)."""
    pdf = gaussian_pdf(mean, cov)
    # contour levels at central-probability quartiles → evenly nested rings
    levels = np.linspace(pdf.max() * 0.05, pdf.max() * 0.95, 6)
    return go.Contour(
        x=GRID, y=GRID, z=pdf,
        showscale=False, contours=dict(start=levels[0], end=levels[-1],
                                       size=levels[1] - levels[0], coloring="lines"),
        line=dict(color=color, width=1.5),
        colorscale=[[0, color], [1, color]],
        showlegend=False, name=label, hoverinfo="skip",
    )
``
# ----------------------------------------------------------------------
# Figure: (a) data | (b) prior | (c) posterior
# ----------------------------------------------------------------------
fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.06)

# (a) data + true mean cross
if n_obs > 0:
    fig.add_trace(go.Scatter(
        x=data[:, 0], y=data[:, 1], mode="markers",
        marker=dict(color="royalblue", size=7),
        name="y_n", hovertemplate="(%{x:.2f}, %{y:.2f})<extra></extra>",
    ), row=1, col=1)
fig.add_trace(go.Scatter(
    x=[Z_TRUE[0]], y=[Z_TRUE[1]], mode="markers",
    marker=dict(symbol="x", color="black", size=14, line=dict(width=2)),
    name="z (true)",
), row=1, col=1)
# (b) prior
fig.add_trace(contour_traces(PRIOR_MEAN, PRIOR_COV, "mediumseagreen", "prior"), row=1, col=2)

# (c) posterior
fig.add_trace(contour_traces(post_mean, post_cov, "crimson", "posterior"), row=1, col=3)
fig.add_trace(go.Scatter(
    x=[post_mean[0]], y=[post_mean[1]], mode="markers",
    marker=dict(symbol="x", color="crimson", size=10), showlegend=False,
), row=1, col=3)

for i, title in enumerate(["(a) data", "(b) prior", f"(c) posterior, N={n_obs}"], start=1):
    fig.layout[f"xaxis{i if i > 1 else ''}"].update(range=[-1, 1], dtick=0.25, constrain="domain")
    fig.layout[f"yaxis{i if i > 1 else ''}"].update(range=[-1, 1], dtick=0.25,
                                                    scaleanchor="x" if i == 1 else f"x{i}",
                                                    scaleratio=1)
fig.update_layout(
    height=380, showlegend=False, plot_bgcolor="white",
    margin=dict(l=10, r=10, t=40, b=10),
    annotations=[
        dict(text="(a) data", showarrow=False, font=dict(size=13), x=0.15, y=1.08, xref="paper", yref="paper"),
        dict(text="(b) prior", showarrow=False, font=dict(size=13), x=0.5, y=1.08, xref="paper", yref="paper"),
        dict(text=f"(c) posterior, N={n_obs}", showarrow=False, font=dict(size=13), x=0.85, y=1.08, xref="paper", yref="paper"),
    ],
)

# ----------------------------------------------------------------------
# Readout
# ----------------------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("knob: σ noise", f"{sigma_noise:.2f}")
c2.metric("posterior mean", f"({post_mean[0]:.2f}, {post_mean[1]:.2f})")
c3.metric("distance to truth", f"{np.linalg.norm(post_mean - Z_TRUE):.3f}")

st.plotly_chart(fig, width="stretch")

st.markdown(
    rf"""
    **How the knob works** — with a conjugate Gaussian model the posterior is exact:

    $$\Sigma_{{post}} = (\Sigma_{{prior}}^{{-1}} + N\,\Sigma_y^{{-1}})^{{-1}}, \quad
      \mu_{{post}} = \Sigma_{{post}}\,(\Sigma_y^{{-1}} \textstyle\sum_n y_n)$$

    Turning **σ** down tightens $\Sigma_y^{{-1}}$, so the posterior snaps toward the data
    mean (high sensitivity). Turning **σ** up flattens the likelihood and the posterior
    collapses back onto the prior (low sensitivity). Increasing **N** has the opposite
    effect of σ — more data always overwhelms the prior, eventually.
    """
)

# ----------------------------------------------------------------------
# Auto-play: animate sigma from small to large, re-rendering the figure
# ----------------------------------------------------------------------
if animate:
    import time
    holder = st.empty()
    for s in np.linspace(0.05, 1.5, frames):
        Sy = (s ** 2) * np.eye(2)
        pc = np.linalg.inv(np.linalg.inv(PRIOR_COV) + n_obs * np.linalg.inv(Sy))
        pm = pc @ (np.linalg.inv(Sy) @ data.sum(axis=0)) if n_obs > 0 else PRIOR_MEAN
        f2 = make_subplots(rows=1, cols=3, horizontal_spacing=0.06)
        if n_obs > 0:
            f2.add_trace(go.Scatter(x=data[:, 0], y=data[:, 1], mode="markers",
                                    marker=dict(color="royalblue", size=6)), row=1, col=1)
        f2.add_trace(go.Scatter(x=[Z_TRUE[0]], y=[Z_TRUE[1]], mode="markers",
                                marker=dict(symbol="x", color="black", size=14)), row=1, col=1)
        f2.add_trace(contour_traces(PRIOR_MEAN, PRIOR_COV, "mediumseagreen", "prior"), row=1, col=2)
        f2.add_trace(contour_traces(pm, pc, "crimson", "posterior"), row=1, col=3)
        f2.update_layout(height=380, showlegend=False,
                         margin=dict(l=10, r=10, t=30, b=10),
                         title=f"knob animating: σ = {s:.2f}")
        for i in range(1, 4):
            xa = f"xaxis{i if i > 1 else ''}"
            ya = f"yaxis{i if i > 1 else ''}"
            f2.layout[xa].update(range=[-1, 1], dtick=0.25, constrain="domain")
            f2.layout[ya].update(range=[-1, 1], dtick=0.25)
        holder.plotly_chart(f2, width="stretch", key=f"anim_{s:.3f}")
        time.sleep(0.12)
