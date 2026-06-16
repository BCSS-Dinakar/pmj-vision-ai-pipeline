# 📸 Reference Data Guide — How to Set Up Your Reference Photos

This guide explains exactly what photos you need to put in the `reference_data/` folder,
how many photos to take, and what makes a good reference photo.

---

## 🤔 What is `reference_data/` and Why Does It Exist?

When the script detects a person on camera, it **cuts out their photo** and tries to figure
out which section they work in. To do this, it compares the person's clothing against the
sample photos you have stored in `reference_data/`.

Think of it like this:
> The AI looks at the person on camera, then looks at your sample photos and says:
> *"This person's uniform looks very similar to the photos in `sec3/`… so this is a sec3 staff member!"*

**If you give it good sample photos → it identifies staff correctly.**
**If you give it bad or too few sample photos → it will label everyone as `customers` by mistake.**

---

## 📁 Folder Structure

Your `reference_data/` folder must look exactly like this:

```
reference_data/
│
├── sec1/          ← Put sample photos of Section 1 staff here
├── sec2/          ← Put sample photos of Section 2 staff here
├── sec3/          ← Put sample photos of Section 3 staff here
├── sec4/          ← Put sample photos of Section 4 staff here
├── sec5/          ← Put sample photos of Section 5 staff here
├── sec6/          ← Put sample photos of Section 6 staff here
├── sec7/          ← Put sample photos of Section 7 staff here
├── sec8/          ← Put sample photos of Section 8 staff here
├── sec9/          ← Put sample photos of Section 9 staff here
└── customers/     ← Put sample photos of regular customers (non-staff) here
```

> ⚠️ Folder names must match **exactly** as shown above (lowercase, no spaces).
> Do NOT rename them. The script looks for these exact names.

---

## 📷 How Many Photos Per Section?

| Folder | Minimum | Recommended | Why |
|---|---|---|---|
| `sec1/` to `sec9/` | 3 photos | **5 to 10 photos** | Covers different people, angles, and lighting |
| `customers/` | 5 photos | **10 to 15 photos** | Covers the wide variety of random casual clothes |

---

## ✅ What Makes a Good Reference Photo?

The AI does NOT only look at color. It also checks:
- **Patterns** (stripes, logos, badges)
- **Texture** (thick jacket, thin t-shirt)
- **Shape** (collar type, sleeve length)
- **Color shading** (how color looks under different lighting)

So your photos should give it a variety of these to learn from.

### ✅ DO — Good photos look like this:
- Clear photo of the full uniform from **chest to waist** (the most identifying part)
- Photo taken in **store lighting** (similar to what the CCTV will see)
- The person is **facing the camera** (front view)
- Photo taken at roughly the **same height** as the CCTV camera angle
- Uniform is **clean and visible** — not covered by a bag or jacket

### ❌ DON'T — Avoid these types of photos:
- Full body photos where the uniform is tiny and hard to see
- Photos taken in very bright sunlight (the store CCTV won't see this)
- Photos where the person is wearing a jacket over their uniform
- Blurry or dark photos
- Screenshots from WhatsApp (heavily compressed, bad quality)
- Photos of just the logo/badge without the full uniform

---

## 📐 Ideal Photo Mix Per Section (5–10 photos)

To get the best results, try to include this mix for each section folder:

| Photo # | What to capture |
|---|---|
| 1 | Person 1 — front view, chest to waist |
| 2 | Person 1 — side view |
| 3 | Person 2 — front view, different body type |
| 4 | Person 3 — front view, under different lighting |
| 5 | Close-up of the uniform collar/logo area |
| 6–10 | More different staff members in the same uniform |

> 💡 **Key Rule:** The photos don't all have to be of the same person.
> As long as they are wearing the same section uniform, add them all!

---

## 👥 Special Guide for `customers/` Folder

The `customers/` folder is different from the section folders.
Customers wear completely random clothes — there is no "uniform" to match.

The AI uses this folder to learn what **non-staff casual clothing** looks like,
so it knows what NOT to classify as a staff member.

**What to put in `customers/`:
- Photos of people wearing regular clothes (t-shirts, shirts, jeans, dresses, etc.)
- Use photos of people with a wide variety of colors and clothing styles
- Include both men and women
- Include different skin tones and body types if possible
- 10 to 15 random photos is enough

> ⚠️ The script also has a built-in safety rule:
> If a person's clothing does NOT match any section above 75% confidence,
> they are **automatically labeled as `customer`** even without a good match
> in the `customers/` folder. But having real customer photos makes this smarter.

---

## 📋 Quick Checklist Before Running the Script

Before you run `python3 main.py`, go through this checklist:

- [ ] All section folders exist inside `reference_data/` (sec1, sec2, ... sec9, customers)
- [ ] Each section folder has **at least 3 photos** (5–10 recommended)
- [ ] Photos show the **full uniform clearly** (not covered, not tiny)
- [ ] Photos are taken in **store-like lighting** (not bright outdoor sunlight)
- [ ] `customers/` folder has at least **5 random people photos**
- [ ] All photos are `.jpg`, `.jpeg`, or `.png` format
- [ ] No random files inside the folders (no PDFs, no Word docs)

---

## 💡 Tips for Best Accuracy

1. **More variety = better accuracy.** 5 different people wearing sec3 uniform > 5 photos of the same one person.
2. **Closer is better.** A photo focused on the chest/torso area of the uniform works better than a tiny full-body shot.
3. **Match the camera angle.** If your CCTV is mounted high and looks down at people, try to take your reference photos from a slightly downward angle too.
4. **Re-collect if accuracy drops.** If after running you notice the AI keeps misclassifying a section, add 2–3 new, clearer photos to that section's folder and run again.
