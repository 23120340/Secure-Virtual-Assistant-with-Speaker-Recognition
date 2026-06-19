#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Find fuzzy-snap landmines: pairs of clean command phrases from DIFFERENT intents
that are close enough (same word count, ratio >= threshold-margin) that a garble of
one could be mis-snapped to the other. These are what the hardened algorithm must
guard against."""
import sys, json, difflib
from pathlib import Path
from collections import defaultdict
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core import asr_corrections as AC
from core import config

TH = config.ASR_FUZZY_THRESHOLD

# snap target -> intent (mirror _load_snap_targets but keep intent)
raw = json.loads((ROOT/"data"/"asr_corrections.json").read_text(encoding="utf-8"))
key_intent = {}      # fuzzy_key -> set of intents
key_canon = {}
for r in raw:
    it = r.get("intent")
    if it in (None, "general_question"):
        continue
    k = AC._fuzzy_key(r["replacement"])
    if not k:
        continue
    key_canon[k] = r["replacement"]
    key_intent.setdefault(k, set()).add(it)

keys = list(key_canon)
print(f"{len(keys)} distinct snap-target keys")

pairs = []
for i in range(len(keys)):
    for j in range(i+1, len(keys)):
        a, b = keys[i], keys[j]
        if len(a.split()) != len(b.split()):
            continue
        ia, ib = key_intent[a], key_intent[b]
        if ia & ib:          # share an intent -> snapping between them is harmless
            continue
        r = difflib.SequenceMatcher(None, a, b).ratio()
        if r >= TH - 0.08:    # near or above threshold = potential landmine
            pairs.append((r, a, b, sorted(ia), sorted(ib)))

pairs.sort(reverse=True)
above = [p for p in pairs if p[0] >= TH]
print(f"\nCROSS-INTENT close pairs (ratio>={TH-0.08:.2f}): {len(pairs)}")
print(f"  of which AT/ABOVE threshold {TH} (active landmines): {len(above)}")
print("\n-- top 40 --")
for r, a, b, ia, ib in pairs[:40]:
    mark = "  <== ACTIVE" if r >= TH else ""
    print(f"  {r:.3f}  {a!r:24} [{','.join(ia)}]  ~  {b!r:24} [{','.join(ib)}]{mark}")
