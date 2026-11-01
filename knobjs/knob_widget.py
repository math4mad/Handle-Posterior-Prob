"""GarageBand-style rotary knob widget for Streamlit.

Frontend gesture math is powered by Knob.js from https://github.com/jherrm/knobs
(MIT-style demo library, zero dependencies). The knob is rendered as SVG inside
a Streamlit Custom Components v2 (CCv2) inline component and reports its value
back to Python via `setStateValue`.
"""

from pathlib import Path

import streamlit as st

_HERE = Path(__file__).parent
_KNOB_LIB_JS = (_HERE / "Knob.js").read_text()

# ----------------------------------------------------------------------
# Component assets (inline CCv2; styles adapt to the Streamlit theme)
# ----------------------------------------------------------------------
_HTML = """
<div class="knob-root">
  <div class="knob-label"></div>
  <div class="knob-dial">
    <svg viewBox="0 0 120 120" width="120" height="120">
      <defs>
        <radialGradient id="knob-body-grad" cx="0.35" cy="0.3" r="0.9">
          <stop offset="0%" stop-color="#5a5f6a"/>
          <stop offset="55%" stop-color="#33363d"/>
          <stop offset="100%" stop-color="#1c1e22"/>
        </radialGradient>
      </defs>
      <path class="knob-track" fill="none" stroke-linecap="round"/>
      <path class="knob-fill" fill="none" stroke-linecap="round"/>
      <circle class="knob-body" cx="60" cy="60" r="42" fill="url(#knob-body-grad)"/>
      <g class="knob-indicator">
        <line x1="10" y1="0" x2="40" y2="0"/>
      </g>
    </svg>
  </div>
  <div class="knob-value"></div>
</div>
<input class="knob-input" type="range"/>
"""

_CSS = """
.knob-root { display: flex; flex-direction: column; align-items: center; gap: 2px; }
.knob-label { color: var(--st-text-color, #333); font-size: 0.85rem; }
.knob-value { color: var(--st-primary-color, #ff4b4b); font-weight: 600;
              font-variant-numeric: tabular-nums; font-size: 0.9rem; }
.knob-dial { position: relative; cursor: grab; user-select: none;
             -webkit-user-select: none; touch-action: none; }
.knob-dial:active { cursor: grabbing; }
.knob-track { stroke: var(--st-secondary-background-color, rgba(128,128,128,0.3)); stroke-width: 5; opacity: 0.9; }
.knob-fill  { stroke: var(--st-primary-color, #ff4b4b); stroke-width: 5; }
.knob-indicator line { stroke: #f2f3f5; stroke-width: 4; stroke-linecap: round; }
.knob-input { display: none; }
"""

# Knob.js source (jherrm/knobs) + our glue code. Knob.js assigns a top-level
# `let Knob`, so concatenation into this module scope works directly.
_JS = _KNOB_LIB_JS + r"""
const SWEEP_START = -135;  // math degrees; valueMax lands here (bottom-left)
const SWEEP_END = 135;     // valueMin lands here (top-left) -> CW increases value

export default function (component) {
  const { data, parentElement, setStateValue } = component;

  const root = parentElement.querySelector(".knob-root");
  const input = parentElement.querySelector(".knob-input");
  const svg = root.querySelector("svg");
  const labelEl = root.querySelector(".knob-label");
  const valueEl = root.querySelector(".knob-value");
  const ind = svg.querySelector(".knob-indicator");
  const fill = svg.querySelector(".knob-fill");
  const track = svg.querySelector(".knob-track");
  const dial = root.querySelector(".knob-dial");

  labelEl.textContent = data.label;

  const CX = 60, CY = 60, R_ARC = 53;
  const rad = (a) => (a * Math.PI) / 180;
  const polar = (a, r) => [CX + r * Math.cos(rad(a)), CY - r * Math.sin(rad(a))];

  // Arc helper. Knob.js maps angleStart->valueMax and angleEnd->valueMin, so a
  // clockwise turn (decreasing math angle) increases the value.
  function arcPath(a0, a1) {
    const [x0, y0] = polar(a0, R_ARC);
    const [x1, y1] = polar(a1, R_ARC);
    const large = Math.abs(a1 - a0) > 180 ? 1 : 0;
    const sweep = a1 < a0 ? 1 : 0; // decreasing math angle == CW on screen
    return `M ${x0.toFixed(2)} ${y0.toFixed(2)} A ${R_ARC} ${R_ARC} 0 ${large} ${sweep} ${x1.toFixed(2)} ${y1.toFixed(2)}`;
  }

  function angleFor(v) {
    const t = (v - data.min) / (data.max - data.min);
    return SWEEP_END + t * (SWEEP_START - SWEEP_END); // valueMin at SWEEP_END
  }

  function drawFill(v) {
    const a = angleFor(v);
    fill.setAttribute("d", arcPath(SWEEP_END, a));
  }

  track.setAttribute("d", arcPath(SWEEP_START, SWEEP_END));

  // ---- instance state survives reruns via the parentElement ----
  if (!dial.__knob) {
    const state = { dragging: false, hydrating: true, handlers: [] };
    dial.__state = state;

    // Knob.js reads min/max/value/angle options off this hidden input.
    // NOTE: value must be set as an *attribute* — the constructor probes it
    // with hasAttribute('value'), a plain property assignment is invisible.
    // step="any" is critical: the default range step=1 snaps values to
    // min + n·step (e.g. min=0.1 turns 1.0 into 1.1).
    input.min = data.min;
    input.max = data.max;
    input.setAttribute("min", data.min);
    input.setAttribute("max", data.max);
    input.setAttribute("step", "any");
    input.setAttribute("value", data.value);
    input.setAttribute("data-angle-start", SWEEP_START);
    input.setAttribute("data-angle-end", SWEEP_END);
    input.setAttribute("data-indicator-auto-rotate", "true");

    const knob = new Knob(input, (k, indicator) => {
      ind.setAttribute(
        "transform",
        `translate(${CX} ${CY}) rotate(${(-indicator.angle).toFixed(2)})`
      );
      const v = parseFloat(k.element.value);
      valueEl.textContent = v.toFixed(data.decimals);
      drawFill(v);
      // Throttle state pushes: coalesce to animation frames and only send
      // when the step-rounded value actually changed. Without this, every
      // intermediate angle triggers a full-app Streamlit rerun and the page
      // visibly jitters while tuning.
      if (state.hydrating) return;
      const q = Math.round(v / data.step) * data.step;
      state.pendingValue = q;
      if (!state.rafPending) {
        state.rafPending = true;
        requestAnimationFrame(() => {
          state.rafPending = false;
          if (state.pendingValue !== state.lastSent) {
            state.lastSent = state.pendingValue;
            setStateValue("value", state.pendingValue);
          }
        });
      }
    });
    dial.__knob = knob;

    // ---- gesture wiring (vanilla JS port of the demo's ui.js setupKnob) ----
    const on = (target, ev, fn, opts) => {
      target.addEventListener(ev, fn, opts);
      state.handlers.push([target, ev, fn, opts]);
    };

    on(dial, "mousedown", (e) => {
      knob.setPosition(dial.offsetLeft, dial.offsetTop);
      knob.doTouchStart([{ pageX: e.pageX, pageY: e.pageY }], e.timeStamp);
      state.dragging = true;
      e.preventDefault();
    });
    on(document, "mousemove", (e) => {
      if (!state.dragging) return;
      knob.doTouchMove([{ pageX: e.pageX, pageY: e.pageY }], e.timeStamp);
      e.preventDefault();
    });
    on(document, "mouseup", () => {
      if (!state.dragging) return;
      knob.doTouchEnd(performance.now());
      state.dragging = false;
    });
    on(dial, "wheel", (e) => {
      knob.setPosition(dial.offsetLeft, dial.offsetTop);
      knob.doMouseScroll(-e.deltaY, e.timeStamp, e.pageX, e.pageY);
      e.preventDefault();
    }, { passive: false });

    if ("ontouchstart" in window) {
      on(dial, "touchstart", (e) => {
        knob.setPosition(dial.offsetLeft, dial.offsetTop);
        knob.doTouchStart(e.targetTouches, e.timeStamp);
        e.preventDefault();
      }, { passive: false });
      on(document, "touchmove", (e) => {
        knob.doTouchMove(e.changedTouches, e.timeStamp);
      });
      on(document, "touchend", () => knob.doTouchEnd(e.timeStamp));
      on(document, "touchcancel", () => knob.doTouchEnd(performance.now()));
    }

    // size + initial publish (setDimensions publishes via __publish)
    knob.setPosition(dial.offsetLeft, dial.offsetTop);
    knob.setDimensions(dial.clientWidth, dial.clientHeight);
    // build-time publishes are suppressed; from here the knob owns its value
    state.hydrating = false;
  }

  // ---- rehydrate on rerun: follow Python state unless a gesture is live ----
  const state = dial.__state;
  const knob = dial.__knob;
  const eps = (data.step || 0) / 2 || 1e-9;
  if (!state.dragging && !state.hydrating && Math.abs(parseFloat(knob.val()) - data.value) > eps) {
    state.hydrating = true;
    knob.val(data.value);
    state.hydrating = false;
  }

  // teardown: drop document-level listeners when Streamlit unmounts us
  return () => {
    for (const [target, ev, fn, opts] of state.handlers) {
      target.removeEventListener(ev, fn, opts);
    }
    delete dial.__knob;
    delete dial.__state;
  };
}
"""

# Declare ONCE at module import time.
_KNOB = st.components.v2.component(
    "rotary_knob",
    html=_HTML,
    css=_CSS,
    js=_JS,
)


def rotary_knob(
    label: str,
    min_value: float,
    max_value: float,
    value: float,
    step: float = 0.1,
    key: str | None = None,
    decimals: int = 2,
):
    """Render a rotary knob and return its current (float) value.

    The value persists in session state under `key` and is updated live while
    the user spins / slides / scrolls the knob.
    """
    # CCv2 stores component state under st.session_state[key] as a (possibly
    # write-through wrapped) mapping {"value": ...}. Reuse it so reruns re-send
    # the knob's own value back to the frontend instead of the initial default.
    current = float(value)
    stored = st.session_state.get(key) if key else None
    if stored is not None and hasattr(stored, "get"):
        try:
            current = float(stored.get("value", current))
        except (TypeError, ValueError):
            pass

    result = _KNOB(
        key=key,
        data={
            "label": label,
            "min": float(min_value),
            "max": float(max_value),
            "value": current,
            "step": float(step),
            "decimals": int(decimals),
        },
        default={"value": current},
        on_value_change=lambda: None,
    )
    return float(result.value)
