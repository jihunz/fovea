"""Synthetic YOLO dataset generator for demos/UI tests.

Usage: python tests/make_synth.py [target_dir]   -> target_dir (default ./synth-fall) + sibling synth-preds/{predA,predB}
Layout: root/images/{train,val,test}, root/labels/{train,val,test}, data.yaml,
plus predictions dirs with confidence for compare-plugin testing."""
import random, os, math
from pathlib import Path
from PIL import Image, ImageDraw

random.seed(7)
ROOT = Path(__import__("sys").argv[1] if len(__import__("sys").argv) > 1 else "./synth-fall")
CLASSES = ["person", "fall_person", "wheelchair"]
COLORS = [(66, 135, 245), (240, 80, 80), (60, 200, 120)]
SPLITS = {"train": 60, "val": 18, "test": 12}

def make_scene(w, h, n_obj):
    img = Image.new("RGB", (w, h), (random.randint(180, 230),)*3)
    d = ImageDraw.Draw(img)
    # background texture
    for _ in range(25):
        x0, y0 = random.randint(0, w), random.randint(0, h)
        d.rectangle([x0, y0, x0+random.randint(10, 80), y0+random.randint(10, 80)], fill=(random.randint(150, 220),)*3)
    boxes = []
    for _ in range(n_obj):
        cls = random.choices([0, 1, 2], weights=[6, 3, 1])[0]
        bw = random.randint(int(w*0.06), int(w*0.3)); bh = random.randint(int(h*0.1), int(h*0.5))
        if cls == 1: bw, bh = bh, bw  # lying person: wide
        x0 = random.randint(0, w-bw); y0 = random.randint(0, h-bh)
        d.rectangle([x0, y0, x0+bw, y0+bh], fill=COLORS[cls], outline=(20, 20, 20), width=2)
        d.ellipse([x0+bw*0.3, y0+bh*0.1, x0+bw*0.7, y0+bh*0.4], fill=(250, 220, 190))
        boxes.append((cls, (x0+bw/2)/w, (y0+bh/2)/h, bw/w, bh/h))
    return img, boxes

def jitter(b, s):
    cls, xc, yc, w, h = b
    return (cls, min(1, max(0, xc+random.uniform(-s, s))), min(1, max(0, yc+random.uniform(-s, s))), max(0.01, w*random.uniform(1-s*3, 1+s*3)), max(0.01, h*random.uniform(1-s*3, 1+s*3)))

idx = 0
for split, n in SPLITS.items():
    (ROOT/"images"/split).mkdir(parents=True, exist_ok=True)
    (ROOT/"labels"/split).mkdir(parents=True, exist_ok=True)
    for pd in ("predA", "predB"):
        (ROOT.parent/"synth-preds"/pd/split).mkdir(parents=True, exist_ok=True)
    for i in range(n):
        seq = f"seq{(idx//10)+1:02d}"
        frame = idx % 10
        name = f"{seq}_cam0_frame_{frame:03d}"
        w, h = random.choice([(640, 480), (1280, 720), (1920, 1080)])
        n_obj = random.choices([0, 1, 2, 3], weights=[1, 5, 3, 1])[0]
        img, boxes = make_scene(w, h, n_obj)
        img.save(ROOT/"images"/split/f"{name}.jpg", quality=85)
        # some images intentionally unlabeled (no label file), some empty label file
        r = random.random()
        if r < 0.08:
            pass  # missing label
        else:
            lines = [f"{c} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}" for c, xc, yc, bw, bh in boxes]
            if r < 0.12 and lines:
                lines[0] = "7 0.5 0.5 1.2 0.3"  # invalid: bad class + out of range
            (ROOT/"labels"/split/f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        # predictions with conf
        for pd, s, miss in (("predA", 0.02, 0.1), ("predB", 0.05, 0.25)):
            plines = []
            for b in boxes:
                if random.random() < miss: continue
                jb = jitter(b, s)
                plines.append(f"{jb[0]} {jb[1]:.6f} {jb[2]:.6f} {jb[3]:.6f} {jb[4]:.6f} {random.uniform(0.3, 0.98):.4f}")
            if random.random() < 0.15:  # false positive
                plines.append(f"{random.randint(0,2)} {random.random():.4f} {random.random():.4f} 0.1 0.2 {random.uniform(0.25,0.6):.4f}")
            (ROOT.parent/"synth-preds"/pd/split/f"{name}.txt").write_text("\n".join(plines) + ("\n" if plines else ""))
        idx += 1

(ROOT/"data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: person\n  1: fall_person\n  2: wheelchair\n")
(ROOT/"README.md").write_text("# synth-fall\nSynthetic fall-detection dataset for Fovea UI tests.\n")
print("done", ROOT)
