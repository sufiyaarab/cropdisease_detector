// Checks that run in the browser before anything is sent to the server.
// The server repeats every check: these exist to give instant, friendly feedback.

export type Check = { ok: true } | { ok: false; message: string };
const pass: Check = { ok: true };
const fail = (message: string): Check => ({ ok: false, message });

export function validateEmail(raw: string): Check {
  const email = raw.trim();
  if (!email) return fail("Enter your email address.");
  if (!/^[^@\s]+@[^@\s]+\.[^@\s]{2,}$/.test(email)) return fail("That email address does not look right.");
  return pass;
}

export function validatePassword(password: string): Check {
  if (password.length < 8) return fail("Use at least 8 characters.");
  if (password.length > 128) return fail("Use 128 characters or fewer.");
  return pass;
}

export function validatePlantName(raw: string): Check {
  const name = raw.trim();
  if (!name) return fail("Give your plant a name.");
  if (name.length > 60) return fail("Use 60 characters or fewer.");
  return pass;
}

const FSA = /^[ABCEGHJ-NPRSTVXY][0-9][ABCEGHJ-NPRSTV-Z]/;

/** Accepts "L1H", "l1h 7k4" or "L1H7K4". Returns the 3-character prefix, the only part we use. */
export function parsePostalCode(raw: string): { ok: true; prefix: string } | { ok: false; message: string } {
  const cleaned = raw.replace(/[\s-]/g, "").toUpperCase();
  if (!cleaned) return { ok: false, message: "Enter your postal code." };
  if (!FSA.test(cleaned) || (cleaned.length !== 3 && cleaned.length !== 6)) {
    return { ok: false, message: "Enter a Canadian postal code, like L1H or L1H 7K4." };
  }
  if (!"KLMNP".includes(cleaned[0])) {
    return { ok: false, message: "Outbreak alerts only cover Ontario postal codes for now." };
  }
  return { ok: true, prefix: cleaned.slice(0, 3) };
}

export const MAX_PHOTOS = 3;
export const MAX_ORIGINAL_BYTES = 25 * 1024 * 1024;
const PHOTO_TYPES = ["image/jpeg", "image/png", "image/webp"];

export function validatePhotoFile(file: { type: string; size: number }): Check {
  if (!PHOTO_TYPES.includes(file.type)) return fail("Use a JPEG, PNG or WebP photo.");
  if (file.size === 0) return fail("This photo file is empty. Take it again.");
  if (file.size > MAX_ORIGINAL_BYTES) return fail("This photo is too large. Take it with the camera button instead.");
  return pass;
}

export interface QualityIssue {
  code: "too_small" | "too_dark" | "too_bright" | "too_blurry";
  message: string;
  tip: string;
}

// Same limits as backend/app/config.py so the phone and the server agree.
export const QUALITY = { minSide: 224, minBrightness: 40, maxBrightness: 225, minSharpness: 40 };

/** Brightness and blur check on greyscale pixels (one value 0-255 per pixel). */
export function assessPixels(gray: ArrayLike<number>, width: number, height: number): QualityIssue[] {
  let sum = 0;
  for (let i = 0; i < gray.length; i++) sum += gray[i];
  const brightness = sum / gray.length;

  // Variance of the Laplacian: few sharp edges means a blurry photo.
  let n = 0;
  let total = 0;
  let totalSq = 0;
  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x;
      const lap = gray[i - width] + gray[i + width] + gray[i - 1] + gray[i + 1] - 4 * gray[i];
      total += lap;
      totalSq += lap * lap;
      n++;
    }
  }
  const sharpness = n ? totalSq / n - (total / n) ** 2 : 0;

  const issues: QualityIssue[] = [];
  if (brightness < QUALITY.minBrightness) {
    issues.push({ code: "too_dark", message: "This photo is too dark.", tip: "Move to brighter light and try again." });
  } else if (brightness > QUALITY.maxBrightness) {
    issues.push({ code: "too_bright", message: "This photo is washed out.", tip: "Avoid direct sun or flash on the leaf." });
  }
  if (sharpness < QUALITY.minSharpness) {
    issues.push({ code: "too_blurry", message: "This photo looks blurry.", tip: "Hold steady and tap the leaf to focus." });
  }
  return issues;
}

export function formatConfidence(value: number): string {
  return `${Math.round(value * 100)}%`;
}
