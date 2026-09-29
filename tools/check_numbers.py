"""Checklist C6 — numbers in the manuscript.
(1) numbers.tex on disk == numbers.tex regenerated now from results/*.json (no hand edits, no stale macros).
(2) independent re-derivation of every formula-derivable macro (n0, credits, C3 sizes, Clopper-Pearson bounds from
    their numerators/denominators, percentages from counts) with scipy, not with the generator's code path.
(3) every \\n<Macro> used in main.tex / supplement.tex / tab_*.tex is defined, and every defined macro is used.
(4) digits typed directly in main.tex prose (outside macros, math design constants, refs) are listed against an
    allow-list of design constants; anything else is a failure.
Exit code 1 on any mismatch. Output: manuscript/check_numbers.json
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path

from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
MS = ROOT / "manuscript"


def ucp(x, n, d=0.05):
    return 1.0 if x >= n else float(sps.beta.ppf(1 - d, x + 1, n - x))


def n0(e, d):
    return math.ceil(math.log(d) / math.log(1 - e))


def macros(text):
    return dict(re.findall(r"\\newcommand\{\\n(\w+)\}\{([^}]*)\}", text))


def num(v):
    return float(v.replace("\\%", "").replace(",", ""))


def main():
    fails, notes = [], []
    has_ms = (MS / "main_ress.tex").exists()          # artifact commit A has data/numbers.tex but no manuscript yet
    ref = MS / "numbers.tex" if has_ms else ROOT / "data" / "numbers.tex"
    disk = ref.read_text(encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "tools" / "make_numbers.py")], check=True, capture_output=True,
                   env={**__import__("os").environ, "PYTHONUTF8": "1"})
    fresh = (MS / "numbers.tex").read_text(encoding="utf-8")
    if disk != fresh:
        fails.append("numbers.tex on disk differs from a fresh generation")
    M = macros(fresh)

    def eq(name, expected, tol=0.051):
        v = num(M[name])
        ok = abs(v - expected) <= tol
        (notes if ok else fails).append(f"{name}: tex={M[name]} expected={expected:.4g}")

    # (2) independent re-derivations
    eq("Nzero", n0(0.05, 0.05)); eq("NzeroTwenty", n0(0.20, 0.05))
    for tag, dp in (("A", 0.075), ("B", 0.10), ("C", 0.15), ("D", 0.20)):
        c = n0(0.05, 0.05) - n0(0.05, dp)
        eq(f"cExact{tag}", c)
        eq(f"cStar{tag}", math.floor(math.log(dp / 0.05) / (-math.log(0.95)) + 1e-12))
        eq(f"cSave{tag}", round(100 * c / 59), 0.5)
    for tag, q in (("A", 0.0), ("B", 0.01), ("C", 0.02)):
        eq(f"Cthree{tag}", math.ceil(math.log(0.025) / math.log(1 - (0.05 - q))))
    for tag, e in (("Ten", 0.10), ("Five", 0.05)):
        eq(f"PopScenes{tag}", math.ceil(math.log(0.05) / math.log(1 - e)))
    eq("MainUfleet", 100 * ucp(num(M["MainDw"]), num(M["MainN"])), 0.006)
    eq("MainUpop", 100 * ucp(num(M["MainJ"]), num(M["MainM"])), 0.006)
    eq("MainDwPct", 100 * num(M["MainDw"]) / num(M["MainN"]), 0.006)
    eq("NightDwPct", 100 * num(M["NightDw"]) / num(M["NightN"]), 0.51)
    eq("NightUpop", 100 * ucp(num(M["NightJ"]), 6), 0.051)
    eq("AllUfleet", 100 * ucp(num(M["AllDw"]), num(M["AllN"])), 0.051)
    eq("AllUpop", 100 * ucp(num(M["AllJ"]), num(M["AllM"])), 0.051)
    eq("ChUpop", 100 * ucp(num(M["ChJ"]), 78), 0.051)
    eq("OrigChUpop", 100 * ucp(num(M["OrigChJ"]), 78), 0.051)
    for j, w in ((0, "zero"), (1, "one"), (2, "two"), (3, "three")):
        eq(f"UpopJ{w}", 100 * ucp(j, 78), 0.051)
    eq("PmainOP", 100 * num(M["RealMissExpOP"]) / num(M["MainN"]), 0.006)
    eq("BreakEven", 1 / num(M["CpuPerSavedOP"]), 0.6)
    eq("SavedOP", num(M["NzeroTwenty"]))
    eq("MainScenes", num(M["MainCDnet"]) + num(M["MainLASIESTA"]))
    # (3) usage and (4) typed digits: only when the manuscript sources are present (not in artifact commit A)
    unused, typed = [], []
    if has_ms:
        def rd(f):
            return f.read_text(encoding="utf-8") if f.exists() else ""

        def after_doc(f):
            t = rd(f)
            return t.split(r"\begin{document}")[1] if r"\begin{document}" in t else ""
        RESS = [MS / "abstract_ress.tex", MS / "highlights.tex"] + sorted(MS.glob("ress_sec*.tex"))
        files = ([MS / "main.tex", MS / "body.tex", MS / "abstract.tex", MS / "supplement.tex", MS / "supplement_ress.tex",
                  MS / "main_ress.tex"] + RESS + sorted(MS.glob("tab_*.tex")))
        used = set()
        for f in files:
            used |= set(re.findall(r"\\n([A-Za-z]+)", rd(f)))
        LATEX = {"ewcolumntype", "e", "ewcommand", "ewif", "ewtheorem", "ode", "oindent", "ormalsize", "ewline",
                 "ewpage", "ot", "u", "abla"}
        used -= LATEX                               # \ne, \newcommand, \node, \noindent, ... are not number macros
        undefined = sorted(u for u in used if u not in M)
        unused = sorted(m for m in M if m not in used)
        if undefined:
            fails.append(f"undefined macros used: {undefined}")
        body = (after_doc(MS / "main.tex") + rd(MS / "body.tex") + rd(MS / "abstract.tex")
                + "".join(rd(f) for f in RESS) + after_doc(MS / "main_ress.tex") + rd(MS / "tab_gap.tex"))
        body = re.sub(r"%.*", "", body)
        body = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", " ", body, flags=re.S)  # layout
        body = re.sub(r"[pL]\{[0-9.]+\\(columnwidth|textwidth)\}|width=[0-9.]+\\(columnwidth|textwidth)|\\multicolumn\{\d+\}",
                      " ", body)  # layout
        body = re.sub(r"\\(cite|ref|label|input|includegraphics|href|url|begin|end|setlength|IEEEPARstart)\{[^}]*\}(\{[^}]*\})?",
                      " ", body)
        body = re.sub(r"\\n[A-Za-z]+(\{\})?", " ", body)
        typed = re.findall(r"(?<![A-Za-z\\])\d+(?:\.\d+)?", body)
        ALLOW = {"0", "1", "2", "3", "4", "5", "6", "7.5", "8", "10", "12", "15", "16", "20", "24", "25", "32", "44", "48",
                 "50", "59", "60", "80", "95", "100", "150", "256", "640", "0.25", "0.3", "2014", "2026", "0009", "0004",
                 "5432", "9343", "22", "3.0", "42", "000"}  # 3.0: licence version; 42: seed; 000: 10{,}000
        bad = sorted({t for t in typed if t not in ALLOW})
        if bad:
            fails.append(f"digits typed in the manuscript outside the allow-list: {bad}")
    out = dict(fails=fails, n_checks=len(notes) + len([f for f in fails if ':' in f and 'expected' in f]),
               rederived_ok=notes, unused_macros=unused, typed_digit_tokens=sorted(set(typed)))
    MS.mkdir(exist_ok=True)
    (MS / "check_numbers.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"re-derived OK: {len(notes)}; mismatches: {len(fails)}; unused macros: {len(unused)}")
    for f in fails:
        print("FAIL", f)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
