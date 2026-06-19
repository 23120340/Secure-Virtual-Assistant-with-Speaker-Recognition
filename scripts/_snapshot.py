#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Snapshot pipeline behavior on a large garble set, classify each result, and
either WRITE a baseline (mode=save) or COMPARE against it (mode=check).

Classification per (source_intent, garble) -> output:
  'right'   : output normalizes to a clean command of the SAME intent  (good)
  'wrong'   : output normalizes to a clean command of a DIFFERENT intent (BAD: cross-intent)
  'garbled' : output unchanged-ish / not a clean command (neutral)
"""
import sys, json, random
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core import asr_corrections as AC
import scripts.generate_asr_corrections as G

SNAP = ROOT / "scripts" / "_baseline.json"

# clean-key -> set(intents)
raw = json.loads((ROOT/"data"/"asr_corrections.json").read_text(encoding="utf-8"))
key_intent = {}
for r in raw:
    it = r.get("intent")
    if it in (None, "general_question"):
        continue
    k = AC._fuzzy_key(r["replacement"])
    if k:
        key_intent.setdefault(k, set()).add(it)
triggers = G.collect_trigger_phrases()
for it, ps in triggers.items():
    for p in ps:
        key_intent.setdefault(AC._fuzzy_key(p), set()).add(it)


def classify(src_intent, garble, out):
    ok = AC._fuzzy_key(out)
    gk = AC._fuzzy_key(garble)
    intents = key_intent.get(ok)
    if intents:
        if src_intent in intents:
            return 'right'
        return 'wrong'
    return 'garbled'


def build_cases():
    rng = random.Random(11)
    cases = []
    for it, ps in triggers.items():
        if it == 'general_question':
            continue
        for phrase in ps:
            for g in G.gen_variants(phrase, rng, 30):
                cases.append((it, phrase, g))
    # plus the known hard garbles outside the generator model (s/x/v + others)
    extra = [
        ('play_music', 'phát nhạc', 'sát nhạc'), ('play_music', 'phát nhạc', 'sáp nhạc'),
        ('play_music', 'phát nhạc', 'xát nhạc'), ('play_music', 'bật nhạc', 'gặt nhạc'),
        ('read_notes', 'xem ghi chú', 'sem ghi chú'), ('read_notes', 'đọc ghi chú', 'đok ghi chú'),
        ('add_note', 'thêm ghi chú', 'thim ghi chú'), ('delete_data', 'xoá ghi chú', 'xoa ghi chú'),
    ]
    cases.extend(extra)
    return cases


def run():
    cases = build_cases()
    out = []
    for src, phrase, g in cases:
        o = AC.apply_corrections(g)
        out.append({'src': src, 'phrase': phrase, 'g': g, 'out': o, 'cls': classify(src, g, o)})
    return out


def summarize(rows):
    from collections import Counter
    c = Counter(r['cls'] for r in rows)
    return c


mode = sys.argv[1] if len(sys.argv) > 1 else 'save'
rows = run()
c = summarize(rows)
print(f"cases={len(rows)}  right={c['right']}  wrong={c['wrong']}  garbled={c['garbled']}")

if mode == 'save':
    SNAP.write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
    print(f"saved baseline -> {SNAP.name}")
    print("\nWRONG (cross-intent) cases in baseline:")
    for r in rows:
        if r['cls'] == 'wrong':
            print(f"  [{r['src']}] {r['g']!r} => {r['out']!r}")
elif mode == 'check':
    base = {(r['src'], r['g']): r for r in json.loads(SNAP.read_text(encoding='utf-8'))}
    regress, fixed, newwrong = [], [], []
    for r in rows:
        b = base.get((r['src'], r['g']))
        if not b:
            continue
        if b['cls'] == 'right' and r['cls'] != 'right':
            regress.append((r, b))       # lost a good correction
        if b['cls'] == 'wrong' and r['cls'] != 'wrong':
            fixed.append((r, b))         # fixed a cross-intent snap
        if b['cls'] != 'wrong' and r['cls'] == 'wrong':
            newwrong.append((r, b))      # introduced a cross-intent snap
    print(f"\nREGRESSIONS (right->not-right): {len(regress)}")
    for r, b in regress[:30]:
        print(f"  [{r['src']}] {r['g']!r}: {b['out']!r} -> {r['out']!r}")
    print(f"\nNEW cross-intent (BAD): {len(newwrong)}")
    for r, b in newwrong[:30]:
        print(f"  [{r['src']}] {r['g']!r}: {b['out']!r} -> {r['out']!r}")
    print(f"\nFIXED cross-intent (good): {len(fixed)}")
    for r, b in fixed[:30]:
        print(f"  [{r['src']}] {r['g']!r}: {b['out']!r} -> {r['out']!r}")
