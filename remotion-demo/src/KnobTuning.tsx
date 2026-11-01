import React from "react";
import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";

// Modern cockpit animation: TWO rotary knobs on a bottom control deck tune a
// Gaussian prior. Stage 1 (0-90f): MU knob slides 1 -> 3. Stage 2 (100-180f):
// SIGMA knob tightens 1 -> 0.4. The active knob glows; the other idles.
// Data truth: mu=3, sigma=2 (dashed line). 30fps, 210 frames, 960x540.

const MU_TRUE = 3;
const X_MIN = -6;
const X_MAX = 11;

const gauss = (x: number, mu: number, sigma: number) =>
  Math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI));

const mapX = (x: number, w: number) =>
  60 + ((x - X_MIN) / (X_MAX - X_MIN)) * (w - 120);
// chart fits between header (~90px) and cockpit deck (starts at 540-178=362)
const SVG_TOP = 88;
const SVG_H = 262;
const Y_MAX_DENSITY = 1.05; // σ=0.4 peaks at ≈1.0 — nothing clips
const mapY = (y: number) => SVG_H - 14 - (y / Y_MAX_DENSITY) * (SVG_H - 52);

// ---------------- single cockpit knob ----------------
const Knob: React.FC<{
  size: number;
  pct: number; // 0..1
  label: string;
  sublabel: string;
  value: number;
  decimals?: number;
  active: boolean;
  accent: string;
}> = ({ size, pct, label, sublabel, value, decimals = 2, active, accent }) => {
  const { fps } = useVideoConfig();
  const frame = useCurrentFrame();
  // sweep: 135deg (min) -> -135deg (max), classic 270-degree knob
  const angle = 135 - pct * 270;
  // glow pulse only while active
  const glow = active
    ? 0.55 + 0.2 * Math.sin(frame / 6 * (Math.PI / fps) * fps)
    : 0;
  const c = size / 2;
  const rTrack = c * 0.78;
  const trackLen = (270 / 360) * 2 * Math.PI * rTrack;
  const circ = 2 * Math.PI * rTrack;
  const filled = ((angle - 135) / -270) * trackLen;

  return (
    <div style={{
      display: "flex", flexDirection: "column", alignItems: "center",
      gap: 4, opacity: active ? 1 : 0.45,
      filter: active ? `drop-shadow(0 0 18px ${accent}66)` : "none",
      transition: "none",
    }}>
      <div style={{ color: "#e8eaee", fontSize: 15, fontWeight: 700, letterSpacing: 1 }}>
        {label}
      </div>
      <div style={{ color: "#9aa0aa", fontSize: 11, letterSpacing: 0.5 }}>{sublabel}</div>
      <svg width={size} height={size}>
        {/* outer ring track (270°) */}
        <circle cx={c} cy={c} r={rTrack} fill="none" stroke="rgba(255,255,255,0.12)"
          strokeWidth={9} strokeLinecap="round"
          strokeDasharray={`${trackLen} ${circ}`} transform={`rotate(135 ${c} ${c})`} />
        {/* filled progress */}
        <circle cx={c} cy={c} r={rTrack} fill="none" stroke={accent}
          strokeWidth={active ? 10 : 8} strokeLinecap="round"
          strokeDasharray={`${filled} ${circ}`} transform={`rotate(135 ${c} ${c})`} />
        {/* tick marks */}
        {Array.from({ length: 11 }).map((_, i) => {
          const a = ((135 - (i / 10) * 270) * Math.PI) / 180;
          const r0 = rTrack - 15, r1 = rTrack - 9;
          return (
            <line key={i}
              x1={c + r0 * Math.cos(a)} y1={c - r0 * Math.sin(a)}
              x2={c + r1 * Math.cos(a)} y2={c - r1 * Math.sin(a)}
              stroke="rgba(255,255,255,0.25)" strokeWidth={1.5} />
          );
        })}
        {/* body */}
        <circle cx={c} cy={c} r={c * 0.56} fill="url(#knobBody)" />
        <circle cx={c} cy={c} r={c * 0.56} fill="none" stroke={active ? accent : "rgba(255,255,255,0.15)"} strokeWidth={1.5} opacity={active ? 0.9 : 1} />
        {/* indicator */}
        <line x1={c} y1={c} x2={c} y2={c - c * 0.46} stroke="#f2f3f5"
          strokeWidth={5} strokeLinecap="round"
          transform={`rotate(${-angle} ${c} ${c})`} />
        {/* ambient glow */}
        <circle cx={c} cy={c} r={c * 0.9} fill="none" stroke={accent} strokeWidth={2} opacity={glow} />
        <defs>
          <radialGradient id="knobBody" cx="0.35" cy="0.3" r="0.95">
            <stop offset="0%" stopColor="#5c626e" />
            <stop offset="60%" stopColor="#2b2e35" />
            <stop offset="100%" stopColor="#17191d" />
          </radialGradient>
        </defs>
      </svg>
      <div style={{
        color: accent, fontSize: 17, fontWeight: 700, fontVariantNumeric: "tabular-nums",
        lineHeight: 1.1,
      }}>
        {value.toFixed(decimals)}
      </div>
    </div>
  );
};

// ---------------- main composition ----------------
export const KnobTuning: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();

  // eased parameter trajectories
  const mu0 = interpolate(frame, [10, 90], [1, 3], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: (t) => spring({ frame: t * 80, fps, config: { damping: 200 } }),
  });
  const sigma0 = interpolate(frame, [100, 180], [1, 0.4], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });
  const stage: 1 | 2 = frame < 95 ? 1 : 2;

  // knob percentages over full widget ranges (mu: -3..7, sigma: 0.1..6)
  const muPct = (mu0 - -3) / 10;
  const sgPct = (sigma0 - 0.1) / 5.9;

  // gaussian curve points
  const pts: string[] = [];
  for (let x = X_MIN; x <= X_MAX; x += 0.1) {
    pts.push(`${mapX(x, width).toFixed(1)},${mapY(gauss(x, mu0, sigma0)).toFixed(1)}`);
  }

  const histBars = [2.1, 2.6, 3.1, 3.6, 4.3, 5.2];
  const histHs = [0.08, 0.13, 0.2, 0.15, 0.09, 0.05];

  return (
    <AbsoluteFill style={{ background: "linear-gradient(180deg, #0d0f13 0%, #14171d 60%, #1a1d24 100%)", fontFamily: "Helvetica, Arial, sans-serif" }}>
      {/* header */}
      <div style={{ display: "flex", alignItems: "baseline", gap: 14, padding: "22px 50px 0" }}>
        <span style={{ color: "#f2f3f5", fontSize: 26, fontWeight: 700 }}>🎛️ Posterior Cockpit</span>
        <span style={{ color: "#8b909a", fontSize: 15 }}>tuning a Gaussian prior · N(μ₀, σ₀²)</span>
      </div>
      <div style={{ color: stage === 1 ? "#ff6b6b" : "#ffb14b", fontSize: 19, padding: "4px 50px", height: 26 }}>
        {stage === 1 ? `stage 1 — localizing: μ₀ = ${mu0.toFixed(2)}` : `stage 2 — envelope: σ₀ = ${sigma0.toFixed(2)}`}
      </div>

      {/* chart */}
      <svg width={width} height={SVG_H} style={{ position: "absolute", top: SVG_TOP }}>
        {/* grid */}
        {[-4, -2, 0, 2, 4, 6, 8, 10].map((gx) => (
          <line key={gx} x1={mapX(gx, width)} y1={8} x2={mapX(gx, width)} y2={SVG_H - 14}
            stroke="rgba(255,255,255,0.05)" strokeWidth={1} />
        ))}
        {/* true mu line */}
        <line x1={mapX(MU_TRUE, width)} y1={8} x2={mapX(MU_TRUE, width)} y2={SVG_H - 14}
          stroke="#f2f3f5" strokeDasharray="6 6" strokeWidth={1.5} opacity={0.55} />
        <text x={mapX(MU_TRUE, width) + 8} y={30} fill="#cfd3d9" fontSize={14}>true μ=3</text>
        {/* data histogram */}
        {histBars.map((cx0, i) => (
          <rect key={i} x={mapX(cx0, width) - 22} width={44}
            y={mapY(histHs[i])} height={SVG_H - 14 - mapY(histHs[i])}
            fill="rgba(100,149,237,0.30)" />
        ))}
        {/* prior curve */}
        <polyline points={pts.join(" ")} fill="none" stroke={stage === 1 ? "#ff4b4b" : "#ffb14b"} strokeWidth={4} />
      </svg>

      {/* cockpit deck */}
      <div style={{
        position: "absolute", bottom: 0, left: 0, right: 0, height: 178,
        background: "linear-gradient(180deg, #232730 0%, #191c22 100%)",
        borderTop: "1px solid rgba(255,255,255,0.08)",
        boxShadow: "0 -12px 40px rgba(0,0,0,0.45)",
        display: "flex", alignItems: "center", justifyContent: "center", gap: 90,
      }}>
        {/* stage indicator */}
        <div style={{
          position: "absolute", left: 50, top: 18,
          display: "flex", gap: 10, alignItems: "center",
        }}>
          {[1, 2].map((s) => (
            <div key={s} style={{
              width: 84, height: 8, borderRadius: 4,
              background: stage === s
                ? (s === 1 ? "#ff4b4b" : "#ffb14b")
                : "rgba(255,255,255,0.1)",
              boxShadow: stage === s ? `0 0 12px ${s === 1 ? "#ff4b4b" : "#ffb14b"}` : "none",
            }} />
          ))}
          <span style={{ color: "#8b909a", fontSize: 13, marginLeft: 6 }}>
            STAGE {stage} / 2
          </span>
        </div>

        <Knob size={112} pct={muPct} label="μ₀" sublabel="prior mean" value={mu0} active={stage === 1} accent="#ff4b4b" />
        <Knob size={112} pct={sgPct} label="σ₀" sublabel="prior σ" value={sigma0} active={stage === 2} accent="#ffb14b" />

        {/* readout panel */}
        <div style={{
          position: "absolute", right: 50, top: "50%", transform: "translateY(-50%)",
          background: "rgba(0,0,0,0.35)", border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: 10, padding: "12px 18px", minWidth: 170,
        }}>
          <div style={{ color: "#8b909a", fontSize: 11, letterSpacing: 1.5, marginBottom: 8 }}>READOUT</div>
          <div style={{ color: "#ff8a8a", fontSize: 17, fontVariantNumeric: "tabular-nums" }}>μ₀ = {mu0.toFixed(2)}</div>
          <div style={{ color: "#ffc77a", fontSize: 17, fontVariantNumeric: "tabular-nums" }}>σ₀ = {sigma0.toFixed(2)}</div>
          <div style={{ color: "#e8eaee", fontSize: 15, marginTop: 6, fontVariantNumeric: "tabular-nums" }}>
            prior: N({mu0.toFixed(2)}, {sigma0.toFixed(2)}²)
          </div>
        </div>
      </div>
    </AbsoluteFill>
  );
};
