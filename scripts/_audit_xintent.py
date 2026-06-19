#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quantify cross-intent corruption: generate realistic Whisper garbles for every
command phrase and report which ones the live pipeline corrects to the WRONG intent
(or fails to correct). Foundation for the fuzzy-snap hardening."""
import sys, json, random
from pathlib import Path
from collections import Counter, defaultdict
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import asr_corrections as AC
from core import config
import scripts.generate_asr_corrections as G

# intent -> set of clean command phrases (triggers + curated cleans)
triggers = G.collect_trigger_phrases()           # intent -> [phrases]
clean_intent = {}                                 # clean phrase -> intent
for it, ps in triggers.items():
    for p in ps:
        clean_intent.setdefault(AC._fuzzy_key(p), it)
# also map every replacement in the file to its intent
raw = json.loads((ROOT/"data"/"asr_corrections.json").read_text(encoding="utf-8"))
for r in raw:
    k = AC._fuzzy_key(r["replacement"])
    if r.get("intent") and r.get("intent") not in ("general_question", None):
        clean_intent.setdefault(k, r["intent"])

rng = random.Random(7)
cross = defaultdict(list)     # (from_intent, to_intent) -> [(garble, got)]
uncorrected = defaultdict(list)
total = 0
for it, ps in triggers.items():
    if it == "general_question":
        continue
    for phrase in ps:
        variants = G.gen_variants(phrase, rng, 40)
        for g in variants:
            total += 1
            out = AC.apply_corrections(g)
            ok_key = AC._fuzzy_key(out)
            out_intent = clean_intent.get(ok_key)
            phrase_key = AC._fuzzy_key(phrase)
            if out_intent and out_intent != it:
                cross[(it, out_intent)].append((g, out))
            elif ok_key == AC._fuzzy_key(g) and ok_key != phrase_key:
                # unchanged garble that isn't already a clean command of its intent
                uncorrected[it].append((g, out))

print(f"generated+tested {total} garbles")
print(f"\n=== CROSS-INTENT corruptions: {sum(len(v) for v in cross.values())} ===")
for (a, b), items in sorted(cross.items(), key=lambda kv: -len(kv[1])):
    print(f"  {a:14} -> {b:14}  : {len(items)}")
    for g, out in items[:4]:
        print(f"        {g!r}  =>  {out!r}")

print(f"\n=== UNCORRECTED (still garbled) by intent: {sum(len(v) for v in uncorrected.values())} ===")
for it, items in sorted(uncorrected.items(), key=lambda kv: -len(kv[1])):
    print(f"  {it:14}: {len(items)}   e.g. {[g for g,_ in items[:5]]}")
