import React from "react";
import {
  AbsoluteFill,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";

// Animation: a rotary knob tunes the parameters of a normal distribution.
// Stage 1 (frames 0-90):  mu0 slides 1 -> 3  (localize the data)
// Stage 2 (frames 90-180): sigma0 tightens 1 -> 0.4 (envelope fine-tune)
// Data truth: mu=3, sigma=2 (dashed line). 30fps, 210 frames.

const MU_TRUE = 3;
const SIGMA_TRUE = 2;
const X_MIN = -6;
const X_MAX = 11;

const gauss = (x: number, mu: number, sigma: number) =>
  Math.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * Math.sqrt(2 * Math.PI));

const mapX = (x: number, w: number) =>
  60 + ((x - X_MIN) / (X_MAX - X_MIN)) * (w - 120);
const SVG_H = 340; // chart svg height (height - 200)
const mapY = (y: number) => SVG_H - 30 - (y / 0.7) * (SVG_H - 60);

export const KnobTuning: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();

  // eased parameter trajectories
  const mu0 = interpolate(frame, [10, 90], [1, 3], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: (t) => spring({ frame: t * 80, fps, config: { damping: 200 } }) / 1,
  });
  const sigma0 = interpolate(frame, [100, 180], [1, 0.4], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
  });

  // knob angle: -135..135 deg mapping value range
  const muPct = (mu0 - -3) / 10;       // mu0 range -3..7
  const sgPct = (sigma0 - 0.1) / 5.9;  // sigma0 range 0.1..6
  const knobAngle = (pct: number) => 135 - pct * 270; // 135deg = min (top-left sweep)
  const stage = frame < 95 ? 1 : 2;
  const angle = stage === 1 ? knobAngle(muPct) : knobAngle(sgPct);

  // gaussian curve points
  const pts: string[] = [];
  for (let x = X_MIN; x <= X_MAX; x += 0.1) {
    pts.push(`${mapX(x, width).toFixed(1)},${mapY(gauss(x, mu0, sigma0)).toFixed(1)}`);
  }

  const label = stage === 1 ? `stage 1 — sliding μ₀ = ${mu0.toFixed(2)}` : `stage 2 — envelope σ₀ = ${sigma0.toFixed(2)}`;

  return (
    <AbsoluteFill style={{ background: "#101216", fontFamily: "Helvetica, Arial, sans-serif" }}>
      {/* title */}
      <div style={{ color: "#f2f3f5", fontSize: 30, fontWeight: 700, padding: "24px 60px" }}>
        🎛️ Posterior Knob — tuning a Gaussian prior
      </div>
      <div style={{ color: "#ff6b6b", fontSize: 22, padding: "0 60px", height: 30 }}>{label}</div>

      {/* knob */}
      <svg width={200} height={200} style={{ position: "absolute", right: 70, top: 110 }}>
        <circle cx={100} cy={100} r={72} fill="none" stroke="rgba(255,255,255,0.15)" strokeWidth={8} strokeLinecap="round"
          strokeDasharray={`${(270 / 360) * 2 * Math.PI * 72} ${2 * Math.PI * 72}`} transform="rotate(135 100 100)" />
        <circle cx={100} cy={100} r={72} fill="none" stroke="#ff4b4b" strokeWidth={8} strokeLinecap="round"
          strokeDasharray={`${(Math.abs(angle - 135) / 360) * 2 * Math.PI * 72} ${2 * Math.PI * 72}`} transform="rotate(135 100 100)" />
        <circle cx={100} cy={100} r={54} fill="url(#g)" />
        <defs>
          <radialGradient id="g" cx="0.35" cy="0.3" r="0.9">
            <stop offset="0%" stopColor="#5a5f6a" />
            <stop offset="100%" stopColor="#1c1e22" />
          </radialGradient>
        </defs>
        <line x1={100} y1={100} x2={100} y2={58} stroke="#f2f3f5" strokeWidth={6} strokeLinecap="round"
          transform={`rotate(${135 - angle + 135} 100 100)`} />
        <text x={100} y={190} textAnchor="middle" fill="#ff6b6b" fontSize={20} fontWeight={700}>
          {(stage === 1 ? mu0 : sigma0).toFixed(2)}
        </text>
      </svg>

      {/* chart */}
      <svg width={width} height={SVG_H} style={{ position: "absolute", top: 180 }}>
        {/* true mu line */}
        <line x1={mapX(MU_TRUE, width)} y1={0} x2={mapX(MU_TRUE, width)} y2={SVG_H}
          stroke="#f2f3f5" strokeDasharray="6 6" strokeWidth={1.5} opacity={0.6} />
        <text x={mapX(MU_TRUE, width) + 8} y={24} fill="#cfd3d9" fontSize={16}>true μ=3</text>
        {/* prior curve */}
        <polyline points={pts.join(" ")} fill="none" stroke="#ff4b4b" strokeWidth={4} />
        {/* fake data histogram as static reference bumps */}
        {[2.1, 2.6, 3.1, 3.6, 4.3, 5.2].map((cx0, i) => {
          const hs = [0.08, 0.13, 0.2, 0.15, 0.09, 0.05][i];
          return (
            <rect key={i} x={mapX(cx0, width) - 22} width={44}
              y={mapY(hs)} height={SVG_H - 30 - mapY(hs)}
              fill="rgba(100,149,237,0.30)" />
          );
        })}
      </svg>
    </AbsoluteFill>
  );
};
