import { describe, expect, it } from "vitest";
import {
  assessPixels,
  formatConfidence,
  parsePostalCode,
  validateEmail,
  validatePassword,
  validatePhotoFile,
  validatePlantName,
} from "./validation";

describe("postal code", () => {
  it("accepts prefixes and full codes in any format", () => {
    for (const input of ["L1H", "l1h 7k4", "L1H7K4", " l1h-7k4 "]) {
      expect(parsePostalCode(input)).toEqual({ ok: true, prefix: "L1H" });
    }
  });
  it("rejects bad input with a message", () => {
    for (const input of ["", "12345", "L1", "L1H 7K", "D1A"]) expect(parsePostalCode(input).ok).toBe(false);
  });
  it("explains that only Ontario is covered", () => {
    const result = parsePostalCode("V6B 1A1");
    expect(result.ok === false && result.message).toContain("Ontario");
  });
});

describe("account fields", () => {
  it("checks email", () => {
    expect(validateEmail("me@example.com").ok).toBe(true);
    expect(validateEmail("").ok).toBe(false);
    expect(validateEmail("me@example").ok).toBe(false);
  });
  it("checks password length", () => {
    expect(validatePassword("12345678").ok).toBe(true);
    expect(validatePassword("1234567").ok).toBe(false);
  });
  it("checks plant name", () => {
    expect(validatePlantName("  Balcony tomato ").ok).toBe(true);
    expect(validatePlantName("   ").ok).toBe(false);
    expect(validatePlantName("x".repeat(61)).ok).toBe(false);
  });
});

describe("photo file", () => {
  it("accepts common photo types", () => {
    expect(validatePhotoFile({ type: "image/jpeg", size: 2_000_000 }).ok).toBe(true);
  });
  it("rejects other files, empty files and huge files", () => {
    expect(validatePhotoFile({ type: "application/pdf", size: 1000 }).ok).toBe(false);
    expect(validatePhotoFile({ type: "image/png", size: 0 }).ok).toBe(false);
    expect(validatePhotoFile({ type: "image/jpeg", size: 30 * 1024 * 1024 }).ok).toBe(false);
  });
});

describe("photo quality", () => {
  const size = 64;
  const image = (pixel: (i: number) => number) => Uint8ClampedArray.from({ length: size * size }, (_, i) => pixel(i));
  const codes = (pixels: Uint8ClampedArray) => assessPixels(pixels, size, size).map((issue) => issue.code);

  it("passes a sharp, well-lit photo", () => {
    expect(codes(image((i) => ((i * 7919) % 97) + 80))).toEqual([]);
  });
  it("flags a flat photo as blurry", () => {
    expect(codes(image(() => 128))).toEqual(["too_blurry"]);
  });
  it("flags dark and bright photos", () => {
    expect(codes(image((i) => (i * 7919) % 20))).toContain("too_dark");
    expect(codes(image((i) => 236 + ((i * 7919) % 20)))).toContain("too_bright");
  });
});

it("formats confidence as a whole percent", () => {
  expect(formatConfidence(0.8349)).toBe("83%");
});
