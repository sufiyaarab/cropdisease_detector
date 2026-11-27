import { describe, expect, it } from "vitest";
import { combine, levelFor, stageIndex } from "./severity";

describe("severity levels", () => {
  it("uses the 5% and 25% cut-offs", () => {
    expect(levelFor(4.9, true)).toBe("mild");
    expect(levelFor(5, true)).toBe("moderate");
    expect(levelFor(24.9, true)).toBe("moderate");
    expect(levelFor(25, true)).toBe("severe");
  });
  it("never says none for a diagnosed problem", () => {
    expect(levelFor(0, true)).toBe("mild");
    expect(levelFor(0, false)).toBe("none");
  });
});

describe("severity calculator", () => {
  it("keeps the higher of photo and answer", () => {
    expect(combine(12, 5, "few")).toEqual({ level: "moderate", leafPct: 12, usedPhoto: true });
    expect(combine(3, 25, "few")).toEqual({ level: "severe", leafPct: 25, usedPhoto: false });
    expect(combine(null, 1, "few")).toEqual({ level: "mild", leafPct: 1, usedPhoto: false });
  });
  it("raises the level when spots spread", () => {
    expect(combine(1, 1, "lower").level).toBe("moderate");
    expect(combine(1, 1, "upper").level).toBe("moderate");
    expect(combine(10, 10, "upper").level).toBe("severe");
    expect(combine(50, 50, "upper").level).toBe("severe");
  });
  it("maps levels to stages", () => {
    expect([stageIndex("mild"), stageIndex("moderate"), stageIndex("severe")]).toEqual([0, 1, 2]);
  });
});
