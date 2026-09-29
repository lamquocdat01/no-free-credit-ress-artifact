"""Count the words of the manuscript abstract with every \\n<Macro> expanded from numbers.tex (math = 1 word)."""
import re
from pathlib import Path

M = Path(__file__).resolve().parents[1] / "manuscript"
nums = dict(re.findall(r"\\newcommand\{\\n(\w+)\}\{([^}]*)\}", (M / "numbers.tex").read_text(encoding="utf-8")))
a = (M / "abstract.tex").read_text(encoding="utf-8").split("\n", 1)[1]   # first line is a comment
a = re.sub(r"\\n([A-Za-z]+)(\{\})?", lambda m: nums.get(m.group(1), "??"), a)
a = re.sub(r"\$[^$]*\$", "X", a)
print(len(a.split()))
print(" ".join(a.split()))
