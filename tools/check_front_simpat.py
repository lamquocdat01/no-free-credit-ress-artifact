"""SIMPAT front-matter gate: abstract word limit (macros expanded, inline math = 1 word), 3-5 highlights of
<= 85 characters each, 1-7 keywords (live GfA 15-10-2026: "1 to 7 keywords"; abstract "does not exceed 250 words"), and venue-language checks on
title/abstract/highlights/keywords: no "RESS", no "reliability demonstration", no "first", no negative-number macro.
ABS_MAX follows the SIMPAT Guide for Authors as captured in notes/GfA_SIMPAT_live_20261015.txt (venue_dossier.md section F2). Exit 1 on failure."""
import re
import sys
from pathlib import Path

ABS_MAX = 250
KW_MAX = 7
M = Path(__file__).resolve().parents[1] / "manuscript"
nums = dict(re.findall(r"\\newcommand\{\\n(\w+)\}\{([^}]*)\}", (M / "numbers.tex").read_text(encoding="utf-8")))


def expand(s):
    s = re.sub(r"\\n([A-Za-z]+)(\{\})?", lambda m: nums.get(m.group(1), "??"), s)
    return s.replace("\\%", "%").replace("--", "-")


fails = []
a = "\n".join(l for l in (M / "abstract_simpat.tex").read_text(encoding="utf-8").splitlines() if not l.startswith("%"))
a = re.sub(r"\$[^$]*\$", "X", expand(a))
nw = len(a.split())
print("abstract words:", nw, f"(limit {ABS_MAX})")
if nw > ABS_MAX or "??" in a:
    fails.append(f"abstract {nw} words")
hl = [expand(l[len("\\item "):].strip()) for l in (M / "highlights_simpat.tex").read_text(encoding="utf-8").splitlines()
      if l.startswith("\\item ")]
for h in hl:
    print(f"{len(h):3d}  {h}")
    if len(h) > 85 or "??" in h:
        fails.append(f"highlight too long ({len(h)}): {h}")
if not 3 <= len(hl) <= 5:
    fails.append(f"{len(hl)} highlights")
front = a + "\n" + "\n".join(hl)
src = (M / "main_simpat.tex").read_text(encoding="utf-8")
k = re.search(r"\\begin\{keyword\}(.*?)\\end\{keyword\}", src, re.S)
kws = [" ".join(x.split()) for x in k.group(1).split("\\sep") if x.strip()] if k else []
print("keywords:", len(kws), kws)
if not 1 <= len(kws) <= KW_MAX:
    fails.append(f"{len(kws)} keywords")
front += "\n" + " ".join(kws)
t = re.search(r"\\newcommand\{\\PaperTitle\}\{(.*?)\}\s*$", (M / "title_simpat.tex").read_text(encoding="utf-8"), re.S)
title = " ".join(t.group(1).split()) if t else ""
print("title:", title)
if not title:
    fails.append("title not found")
front += "\n" + title
for pat in (r"\bRESS\b", r"reliability\s+demonstration", r"\bfirst\b"):
    if re.search(pat, front, re.I):
        fails.append(f"front matter contains /{pat}/")
neg = [k for k, v in nums.items() if v.strip().startswith("-") or v.strip().startswith("$-")]
if neg:
    fails.append(f"negative macros: {neg}")
for f in fails:
    print("FAIL", f)
sys.exit(1 if fails else 0)
