// Drawn leaves for the disease pictures. Drawings, not photos: no licence issues,
// and they show one clear pattern per disease. Real photos replace them only when
// their licence allows reuse (see docs/disease-visuals.md).
import type { ReactNode } from "react";
import type { LeafLook } from "../api/types";

const LEAF_PATH: Record<LeafLook["leaf"], string> = {
  tomato: "M50 136 L50 112 C18 102 4 72 9 42 C14 16 34 4 50 4 C66 4 86 16 91 42 C96 72 82 102 50 112 Z",
  apple: "M50 136 L50 118 C22 110 10 80 12 54 C14 26 34 8 50 4 C66 8 86 26 88 54 C90 80 78 110 50 118 Z",
  corn: "M50 138 C34 120 32 60 38 20 C42 8 48 3 50 2 C52 3 58 8 62 20 C68 60 66 120 50 138 Z",
};

/** Where spots may go on each leaf, and that area in viewBox units (used for the coverage). */
const INSIDE: Record<LeafLook["leaf"], { test: (x: number, y: number) => boolean; area: number }> = {
  tomato: { test: (x, y) => ((x - 50) / 34) ** 2 + ((y - 60) / 44) ** 2 < 1, area: 6700 },
  apple: { test: (x, y) => ((x - 50) / 32) ** 2 + ((y - 62) / 46) ** 2 < 1, area: 6500 },
  corn: { test: (x, y) => Math.abs(x - 50) < 10 && y > 34 && y < 118, area: 3400 },
};

/** Approximate area of one lesion divided by r squared, per lesion type. */
const AREA_FACTOR: Record<string, number> = {
  ring: 5.7, blotch: 4.1, smallspot: 3.1, stipple: 3.1, velvet: 3.1, rust: 4.5,
  frogeye: 3.1, cigar: 3.1, pustule: 0.8, rect: 1.4,
};
const MANY = new Set(["smallspot", "stipple", "pustule"]);
const LONG = new Set(["cigar", "rect"]);

/** Main colour of each lesion type, for the small plant drawing. */
const LESION_COLOUR: Record<string, string> = {
  ring: "#6e4b2a", blotch: "#4a3b2a", smallspot: "#5a3a22", stipple: "#d9d29a", velvet: "#4f4d26",
  rust: "#e08a1e", frogeye: "#7b3f6e", cigar: "#9c916e", pustule: "#a0521e", rect: "#8d8774", none: "#5f9a4c",
};

function mix(a: string, b: string, t: number): string {
  const pa = [1, 3, 5].map((i) => parseInt(a.slice(i, i + 2), 16));
  const pb = [1, 3, 5].map((i) => parseInt(b.slice(i, i + 2), 16));
  return "#" + pa.map((v, i) => Math.round(v + (pb[i] - v) * t).toString(16).padStart(2, "0")).join("");
}

/** Small seeded random generator so a drawing looks the same every time. */
function random(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function Lesion({ type, x, y, r }: { type: string; x: number; y: number; r: number }): ReactNode {
  switch (type) {
    case "ring":
      return (
        <g>
          <circle cx={x} cy={y} r={r * 1.35} fill="#d8c75a" />
          <circle cx={x} cy={y} r={r} fill="#6e4b2a" />
          <circle cx={x} cy={y} r={r * 0.68} fill="none" stroke="#3d2914" strokeWidth={Math.max(0.5, r * 0.12)} />
          <circle cx={x} cy={y} r={r * 0.36} fill="none" stroke="#3d2914" strokeWidth={Math.max(0.5, r * 0.12)} />
        </g>
      );
    case "blotch":
      return (
        <g>
          <ellipse cx={x} cy={y} rx={r * 1.5} ry={r * 1.15} fill="#a9c27a" transform={`rotate(25 ${x} ${y})`} />
          <ellipse cx={x} cy={y} rx={r * 1.25} ry={r * 0.95} fill="#4a3b2a" transform={`rotate(25 ${x} ${y})`} />
        </g>
      );
    case "smallspot":
      return (
        <g>
          <circle cx={x} cy={y} r={r} fill="#d9d2c0" stroke="#4a2e1a" strokeWidth={r * 0.4} />
          <circle cx={x} cy={y} r={r * 0.22} fill="#1c2420" />
        </g>
      );
    case "stipple":
      return (
        <g fill="#efe9b8">
          {[[0, 0], [0.9, 0.3], [-0.7, 0.6], [0.3, -0.9], [-0.5, -0.5], [0.6, 0.9]].map(([dx, dy], i) => (
            <circle key={i} cx={x + dx * r} cy={y + dy * r} r={Math.max(0.5, r * 0.22)} />
          ))}
        </g>
      );
    case "velvet":
      return <circle cx={x} cy={y} r={r} fill="#4f4d26" stroke="#7a7a3a" strokeWidth={r * 0.3} strokeDasharray={`${r * 0.3} ${r * 0.2}`} />;
    case "rust":
      return (
        <g>
          <circle cx={x} cy={y} r={r * 1.2} fill="#b3261e" />
          <circle cx={x} cy={y} r={r} fill="#f0a830" />
          <circle cx={x} cy={y} r={r * 0.25} fill="#a0521e" />
        </g>
      );
    case "frogeye":
      return (
        <g>
          <circle cx={x} cy={y} r={r} fill="#7b3f6e" />
          <circle cx={x} cy={y} r={r * 0.7} fill="#c8a97a" />
        </g>
      );
    case "cigar":
      return <ellipse cx={x} cy={y} rx={r * 0.45} ry={r * 2.2} fill="#b9ad8a" stroke="#7d7255" strokeWidth={r * 0.12} />;
    case "pustule":
      return <ellipse cx={x} cy={y} rx={r * 0.35} ry={r * 0.7} fill="#a0521e" />;
    case "rect":
      return <rect x={x - r * 0.3} y={y - r * 1.2} width={r * 0.6} height={r * 2.4} fill="#a9a390" stroke="#6e6a58" strokeWidth={r * 0.1} />;
    default:
      return null;
  }
}

interface LeafProps {
  look: Pick<LeafLook, "leaf" | "lesion">;
  /** Share of the leaf covered by spots, 0 to 1. */
  amount: number;
  /** 0 = green, 1 = fully yellowed. */
  yellowing?: number;
  width?: number;
  seed?: number;
  label: string;
}

/** A whole leaf with spots covering about `amount` of it. */
export function LeafDrawing({ look, amount, yellowing = 0, width = 72, seed = 7, label }: LeafProps) {
  const inside = INSIDE[look.leaf];
  const spots: { x: number; y: number; r: number }[] = [];
  if (look.lesion !== "none" && amount > 0) {
    const long = LONG.has(look.lesion); // long thin spots must stay inside a narrow leaf
    const n = MANY.has(look.lesion) ? Math.min(40, 4 + Math.round(amount * 60)) : Math.min(long ? 16 : 10, 1 + Math.round(amount * (long ? 24 : 14)));
    const r = Math.min(long ? 5.5 : 16, Math.max(1.6, Math.sqrt((amount * inside.area) / (n * (AREA_FACTOR[look.lesion] ?? 3.1)))));
    const rand = random(seed);
    for (let i = 0; i < n; i++) {
      let best = { x: 50, y: 60 };
      for (let tries = 0; tries < 120; tries++) {
        const x = 14 + rand() * 72;
        const y = 10 + rand() * 120;
        if (!inside.test(x, y)) continue;
        best = { x, y };
        if (spots.every((s) => Math.hypot(s.x - x, s.y - y) > r * 1.9)) break;
      }
      spots.push({ ...best, r });
    }
  }
  const fill = mix("#5f9a4c", "#c9b54a", yellowing);
  const vein = mix("#3f7232", "#7a6a2a", yellowing);
  return (
    <svg width={width} height={width * 1.4} viewBox="0 0 100 140" role={label ? "img" : undefined} aria-label={label || undefined} aria-hidden={label ? undefined : true}>
      <path d={LEAF_PATH[look.leaf]} fill={fill} stroke={vein} strokeWidth="1.5" />
      {look.leaf === "corn" ? (
        <path d="M50 136 L50 6 M44 124 L43 26 M56 124 L57 26" stroke={vein} strokeWidth="0.8" fill="none" />
      ) : (
        <path d="M50 112 L50 10 M50 40 L30 26 M50 40 L70 26 M50 62 L24 48 M50 62 L76 48 M50 84 L28 74 M50 84 L72 74" stroke={vein} strokeWidth="1" fill="none" />
      )}
      {spots.map((s, i) => <Lesion key={i} type={look.lesion} x={s.x} y={s.y} r={s.r} />)}
    </svg>
  );
}

/** One spot, magnified. */
export function LesionCloseup({ lesion, width = 72, label }: { lesion: string; width?: number; label: string }) {
  const r = lesion === "cigar" || lesion === "rect" || lesion === "pustule" ? 12 : MANY.has(lesion) ? 9 : 13;
  return (
    <svg width={width} height={width} viewBox="0 0 40 40" role="img" aria-label={label}>
      <circle cx="20" cy="20" r="19" fill="#5f9a4c" />
      {lesion === "none" ? (
        <path d="M20 4 L20 36 M20 14 L10 8 M20 14 L30 8 M20 24 L9 18 M20 24 L31 18" stroke="#3f7232" strokeWidth="1" fill="none" />
      ) : (
        <Lesion type={lesion} x={20} y={20} r={lesion === "pustule" ? 14 : r} />
      )}
    </svg>
  );
}

/** A small plant showing which leaves the problem shows up on first. */
export function PlantDrawing({ look, width = 64, label }: { look: LeafLook; width?: number; label: string }) {
  // Leaves from the top down: [x, y]
  const leaves = [[30, 20], [70, 34], [30, 50], [70, 66], [30, 82]];
  const hit = (i: number) =>
    look.lesion !== "none" && (look.where === "all" ? i % 2 === 1 || i === 4 : look.where === "lower" ? i >= 3 : i <= 1);
  const colour = LESION_COLOUR[look.lesion] ?? "#6e4b2a";
  return (
    <svg width={width} height={width} viewBox="0 0 100 100" role="img" aria-label={label}>
      <path d="M50 98 L50 6" stroke="#3f7232" strokeWidth="3" />
      {leaves.map(([x, y], i) => (
        <g key={i}>
          <ellipse cx={x} cy={y} rx="18" ry="7" fill={hit(i) ? "#a39a45" : "#5f9a4c"} />
          {hit(i) && (
            <>
              <circle cx={x - 6} cy={y} r="3" fill={colour} />
              <circle cx={x + 5} cy={y - 1} r="2.5" fill={colour} />
            </>
          )}
        </g>
      ))}
    </svg>
  );
}
