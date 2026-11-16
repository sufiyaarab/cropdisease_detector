// Prepares a photo on the phone before upload:
//   1. checks the file type and size
//   2. shrinks it to about 1 MB
//   3. redraws it on a canvas, which removes GPS and all other hidden photo data
//   4. checks brightness and blur so the user can retake it right away
import { assessPixels, QUALITY, validatePhotoFile, type QualityIssue } from "./validation";

const MAX_SIDE = 1600;
const TARGET_BYTES = 1_000_000;

export interface PreparedPhoto {
  blob: Blob | null; // null when the photo cannot be used
  previewUrl: string | null;
  issues: QualityIssue[] | { code: string; message: string; tip: string }[];
}

function toBlob(canvas: HTMLCanvasElement, quality: number): Promise<Blob | null> {
  return new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", quality));
}

export async function preparePhoto(file: File): Promise<PreparedPhoto> {
  const fileCheck = validatePhotoFile(file);
  if (!fileCheck.ok) {
    return { blob: null, previewUrl: null, issues: [{ code: "bad_file", message: fileCheck.message, tip: "" }] };
  }

  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
  } catch {
    const message = "We could not open this photo.";
    return { blob: null, previewUrl: null, issues: [{ code: "unreadable_image", message, tip: "Take a new photo." }] };
  }

  if (Math.min(bitmap.width, bitmap.height) < QUALITY.minSide) {
    bitmap.close();
    const issue = { code: "too_small", message: "This photo is too small.", tip: "Move closer or use the camera button." };
    return { blob: null, previewUrl: null, issues: [issue] };
  }

  const scale = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(bitmap.width * scale);
  canvas.height = Math.round(bitmap.height * scale);
  canvas.getContext("2d")!.drawImage(bitmap, 0, 0, canvas.width, canvas.height);

  // Quality check on a small greyscale copy.
  const checkScale = Math.min(1, 512 / Math.max(canvas.width, canvas.height));
  const small = document.createElement("canvas");
  small.width = Math.max(3, Math.round(canvas.width * checkScale));
  small.height = Math.max(3, Math.round(canvas.height * checkScale));
  const ctx = small.getContext("2d", { willReadFrequently: true })!;
  ctx.drawImage(canvas, 0, 0, small.width, small.height);
  const rgba = ctx.getImageData(0, 0, small.width, small.height).data;
  const gray = new Uint8ClampedArray(small.width * small.height);
  for (let i = 0; i < gray.length; i++) {
    gray[i] = 0.299 * rgba[i * 4] + 0.587 * rgba[i * 4 + 1] + 0.114 * rgba[i * 4 + 2];
  }
  const issues = assessPixels(gray, small.width, small.height);
  bitmap.close();

  let blob: Blob | null = null;
  for (const quality of [0.85, 0.75, 0.65, 0.5]) {
    blob = await toBlob(canvas, quality);
    if (blob && blob.size <= TARGET_BYTES) break;
  }
  if (!blob) {
    const message = "We could not process this photo.";
    return { blob: null, previewUrl: null, issues: [{ code: "unreadable_image", message, tip: "Take a new photo." }] };
  }
  return { blob: issues.length ? null : blob, previewUrl: URL.createObjectURL(blob), issues };
}
