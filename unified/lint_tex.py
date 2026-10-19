"""
Structural lint for the assembled manuscript.

Substitutes for a local LaTeX compile (no TeX toolchain on this machine):
checks environment balance, label/ref consistency, citation keys against the
bibliography, duplicate labels, stray placeholders and missing figure files.

    python lint_tex.py ../../main_revised.tex
"""
from __future__ import annotations

import os
import re
import sys
from collections import Counter


def strip_comments(text: str) -> str:
    out = []
    for line in text.split("\n"):
        i, esc = None, False
        for j, ch in enumerate(line):
            if ch == "\\":
                esc = not esc
            elif ch == "%" and not esc:
                i = j
                break
            else:
                esc = False
        out.append(line if i is None else line[:i])
    return "\n".join(out)


def main(path: str) -> int:
    raw = open(path, encoding="utf-8").read()
    text = strip_comments(raw)
    root = os.path.dirname(os.path.abspath(path))
    problems, warnings = [], []

    # 1. environment balance
    stack = []
    for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", text):
        kind, name = m.group(1), m.group(2)
        line = text[:m.start()].count("\n") + 1
        if kind == "begin":
            stack.append((name, line))
        else:
            if not stack:
                problems.append(f"line {line}: \\end{{{name}}} with no matching begin")
            elif stack[-1][0] != name:
                problems.append(
                    f"line {line}: \\end{{{name}}} closes \\begin{{{stack[-1][0]}}} "
                    f"opened at line {stack[-1][1]}")
                stack.pop()
            else:
                stack.pop()
    for name, line in stack:
        problems.append(f"line {line}: \\begin{{{name}}} never closed")

    # 2. labels and refs
    labels = re.findall(r"\\label\{([^}]+)\}", text)
    dupes = [k for k, v in Counter(labels).items() if v > 1]
    for d in dupes:
        problems.append(f"duplicate label: {d}")
    refs = set(re.findall(r"\\(?:ref|autoref|eqref)\{([^}]+)\}", text))
    for r in sorted(refs - set(labels)):
        problems.append(f"undefined reference: \\ref{{{r}}}")
    unused = set(labels) - refs
    for u in sorted(unused):
        warnings.append(f"label never referenced: {u}")

    # 3. citations
    bibkeys = set(re.findall(r"\\bibitem\{([^}]+)\}", text))
    cited = set()
    for m in re.finditer(r"\\cite\{([^}]+)\}", text):
        cited |= {k.strip() for k in m.group(1).split(",")}
    for c in sorted(cited - bibkeys):
        problems.append(f"citation with no bibitem: \\cite{{{c}}}")
    for b in sorted(bibkeys - cited):
        warnings.append(f"bibitem never cited: {b}")

    # 4. placeholders left behind
    for pat in [r"@@[A-Z_]+@@", r"\\FILL\{", r"\[model\]", r"\[auc\]"]:
        for m in re.finditer(pat, text):
            line = text[:m.start()].count("\n") + 1
            problems.append(f"line {line}: unfilled placeholder {m.group(0)!r}")

    # 5. figures present on disk
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text):
        fn = m.group(1)
        line = text[:m.start()].count("\n") + 1
        cands = [os.path.join(root, fn)]
        if not os.path.splitext(fn)[1]:
            cands = [os.path.join(root, fn + e)
                     for e in (".png", ".pdf", ".jpg", ".jpeg", ".eps")]
        if not any(os.path.exists(c) for c in cands):
            problems.append(f"line {line}: missing figure file {fn}")

    # 6. report
    print(f"lint: {os.path.basename(path)}  "
          f"({len(text.splitlines())} lines, {len(bibkeys)} references, "
          f"{len(cited)} cited, {len(labels)} labels)")
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"  ERROR  {p}")
    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for w in warnings[:25]:
            print(f"  warn   {w}")
        if len(warnings) > 25:
            print(f"  ... and {len(warnings)-25} more")
    if not problems:
        print("\nno structural errors found")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "main.tex"))
