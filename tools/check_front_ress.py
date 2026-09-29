"""RESS front-matter gate: abstract <= 200 words (macros expanded, inline math = 1 word), 3-5 highlights of
<= 85 characters each (including spaces, macros expanded), 1-7 keywords. Exit 1 on failure."""
import re
import sys
from pathlib import Path

M = Path(__file__).resolve().parents[1] / "manuscript"
nums = dict(re.findall(r"\\newcommand\{\\n(\w+)\}\{([^}]*)\}", (M / "numbers.tex").read_text(encoding="utf-8")))


def expand(s):
    s = re.sub(r"\\n([A-Za-z]+)(\{\})?", lambda m: nums.get(m.group(1), "??"), s)
    return s.replace("\\%", "%")


fails = []
a = "\n".join(l for l in (M / "abstract_ress.tex").read_text(encoding="utf-8").splitlines() if not l.startswith("%"))
a = re.sub(r"\$[^$]*\$", "X", expand(a))
nw = len(a.split())
print("abstract words:", nw)
if nw > 200 or "??" in a:
    fails.append(f"abstract {nw} words")
hl = [expand(l[len("\\item "):].strip()) for l in (M / "highlights.tex").read_text(encoding="utf-8").splitlines()
      if l.startswith("\\item ")]
for h in hl:
    print(f"{len(h):3d}  {h}")
    if len(h) > 85 or "??" in h:
        fails.append(f"highlight too long ({len(h)}): {h}")
if not 3 <= len(hl) <= 5:
    fails.append(f"{len(hl)} highlights")
kw_file = M / "main_ress.tex"
if kw_file.exists():
    k = re.search(r"\\begin\{keyword\}(.*?)\\end\{keyword\}", kw_file.read_text(encoding="utf-8"), re.S)
    if k:
        kws = [x.strip() for x in k.group(1).split("\\sep") if x.strip()]
        print("keywords:", len(kws), kws)
        if not 1 <= len(kws) <= 7:
            fails.append(f"{len(kws)} keywords")
for f in fails:
    print("FAIL", f)
sys.exit(1 if fails else 0)
