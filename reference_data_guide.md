# 📸 Reference Data Guide — How to Set Up Your Reference Photos

Explains what photos to put in `reference_data/`, how many, and what makes a good reference image for the clothing matcher.

---

## 🤔 What is `reference_data/` and Why Does It Exist?

When the pipeline detects a person in a camera frame, it:
1. Cuts out a **transparent RGBA crop** of just the person (background removed by YOLO mask)
2. Computes a **ResNet18 embedding** (captures clothing texture and shape)
3. Computes an **HSV color histogram** (captures dominant clothing color)
4. Compares both against the pre-computed fingerprints for each image in `reference_data/`
5. The combined score (60% ResNet + 40% color) determines the section label

> **Good reference photos → correct section labels.**
> **Blurry, few, or wrong photos → everyone gets labeled as `customers`.**

The computed fingerprints are cached in `reference_cache.pkl` on first run. If you add or change reference photos, delete this file so the cache rebuilds.

---

## 📁 Folder Structure

The folder names must exactly match the keys in `CLASS_MAPPING` inside `main.py`:

```
reference_data/
│
├── sec1/              ← Section 1 staff uniform photos
├── sec1-sec2-sec3/    ← Combined zone photos (if applicable)
├── sec2/
├── sec3/
├── sec4/
├── sec5/
├── sec6/
├── sec7/
├── sec8/
├── sec10/
└── customers/         ← Regular customer / non-staff photos
```

> ⚠️ Folder names are **case-sensitive** and must match exactly. If a folder doesn't exist in `CLASS_MAPPING`, it is silently ignored.

---

## 📷 How Many Photos Per Section?

| Folder | Minimum | Recommended |
|---|---|---|
| `sec1/` … `sec10/` | 3 | **5–10** — different people, angles, lighting |
| `customers/` | 5 | **10–15** — wide variety of random casual clothing |

---

## ✅ What Makes a Good Reference Photo?

The matcher uses **two signals** — shape/texture (ResNet) and color (HSV histogram). Good photos cover both:

### ✅ DO:
- Clear photo of the uniform, **chest to waist** (most identifying area)
- Photo taken in **store / indoor lighting** (similar to what the CCTV sees)
- Person is **facing the camera** (front view preferred)
- Uniform is **fully visible** — not covered by a bag, jacket, or arm
- Ideally **cropped close to the person** rather than a wide full-room shot
- **Transparent PNG** cutouts (from `segment_dataset.py`) work best since the background is already removed

### ❌ DON'T:
- Full-body distant shots where the uniform is too small
- Bright outdoor or sunlit photos (very different lighting from CCTV)
- Person wearing a jacket or coat over the uniform
- Blurry or very dark photos
- WhatsApp-compressed screenshots (artifacts hurt ResNet)
- Photos of just the logo or badge — the model needs the full clothing area

---

## 📐 Suggested Photo Mix Per Section (5–10 photos)

| # | Capture |
|---|---|
| 1 | Person A — front, chest to waist |
| 2 | Person A — slight side angle |
| 3 | Person B — front, different body type |
| 4 | Person C — under different lighting |
| 5 | Close-up of collar / logo / badge area |
| 6–10 | Additional staff members in the same uniform |

> 💡 **More variety = better accuracy.** Five different staff members in the sec3 uniform beats five photos of the same person.

---

## 👥 Special Guide for `customers/`

Customers have no uniform — the AI uses this folder to learn what **non-staff casual clothing** looks like so it knows what NOT to classify as staff.

- Wide variety of colors, styles, fabrics — t-shirts, shirts, jeans, dresses
- Include both men and women, different body types
- 10–15 photos is sufficient

> ℹ️ Built-in safety rule: if no reference match exceeds the `threshold` (default `0.75`), the person is automatically labeled `customers` regardless of the `customers/` folder content. Good `customers/` photos improve the color histogram comparison, making borderline cases more accurate.

---

## 🔄 Rebuilding the Cache

On first run, the pipeline processes every reference image through ResNet18 and saves the result to `reference_cache.pkl`. This makes subsequent runs start instantly.

**Whenever you add, remove, or change reference photos — delete the cache file:**

```bash
rm reference_cache.pkl
```

Then re-run `python3 main.py` and the cache will rebuild automatically.

---

## 📋 Checklist Before Running

- [ ] All section folders exist inside `reference_data/` with names matching `CLASS_MAPPING`
- [ ] Each section has **at least 3 photos** (5–10 recommended)
- [ ] Photos show the **full uniform clearly** — not covered, not tiny
- [ ] Photos taken in **store/indoor lighting**
- [ ] `customers/` has at least **5 diverse casual clothing photos**
- [ ] All photos are `.jpg`, `.jpeg`, or `.png` (no PDFs, docs, etc.)
- [ ] If you changed any reference photos, `reference_cache.pkl` has been deleted

---

## 💡 Tips for Best Accuracy

1. **More people, same uniform > more photos of one person.** Diversity helps ResNet generalize.
2. **Closer crops are better.** The pipeline crops the inner 80% of each reference image before embedding — so a tight crop of the torso is more effective than a wide room shot.
3. **Match the CCTV angle.** If your camera is high and looks down at people, try to take reference photos from a slightly downward angle too.
4. **Use transparent PNGs from `segment_dataset.py`.** Running your reference photos through the segmentation tool first removes the background, so the ResNet and color histogram only see the clothing — exactly what the live pipeline sees.
5. **Accuracy still low after adding photos?** Lower the `threshold` in `main.py` from `0.75` toward `0.65` to accept softer matches.
