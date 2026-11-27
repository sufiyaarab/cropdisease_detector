// Severity levels and the severity calculator's rule.
// The level cut-offs match backend/app/pipeline/severity.py.
import type { SeverityLevel } from "../api/types";

export const LEVELS: SeverityLevel[] = ["none", "mild", "moderate", "severe"];
export const LEVEL_NAME: Record<SeverityLevel, string> = { none: "None", mild: "Mild", moderate: "Moderate", severe: "Severe" };

/** Share of the leaf with spots (%) -> level. A diagnosed problem is never "none". */
export function levelFor(leafPct: number, hasDisease: boolean): SeverityLevel {
  if (!hasDisease && leafPct < 5) return "none";
  if (leafPct < 5) return "mild";
  if (leafPct < 25) return "moderate";
  return "severe";
}

/** The reference leaves the user picks from in the calculator (% of leaf with spots). */
export const REFERENCE_PCTS = [1, 5, 10, 25, 50] as const;

export type Spread = "few" | "lower" | "upper";
export const SPREAD_TEXT: Record<Spread, string> = {
  few: "Just 1 or 2 leaves",
  lower: "Several lower leaves",
  upper: "Lower and upper leaves",
};

export interface Combined {
  level: SeverityLevel;
  leafPct: number;
  usedPhoto: boolean;
}

/**
 * Combines the photo estimate with the user's answers. When unsure it rounds UP,
 * so a problem is not missed: it keeps the higher leaf share, and spread across
 * the plant raises the level.
 */
export function combine(photoPct: number | null, userPct: number, spread: Spread): Combined {
  const usedPhoto = photoPct !== null && photoPct > userPct;
  const leafPct = usedPhoto ? photoPct : userPct;
  let index = LEVELS.indexOf(levelFor(leafPct, true));
  if (spread === "lower") index = Math.max(index, 2);
  if (spread === "upper") index = Math.min(3, index + 1);
  return { level: LEVELS[index], leafPct, usedPhoto };
}

/** Which of the three stages (early, spreading, severe) a level points to. */
export function stageIndex(level: SeverityLevel): number {
  return level === "severe" ? 2 : level === "moderate" ? 1 : 0;
}

export const LEVEL_ADVICE: Record<SeverityLevel, string> = {
  none: "No spots found. Keep checking your plants every week.",
  mild: "Caught early. Follow the care plan now and take a new photo in about 7 days.",
  moderate: "It is spreading. Follow the care plan this week and take a new photo in 5 to 7 days.",
  severe: "Act today. Follow the care plan, and show the plant to a garden centre or plant expert.",
};
