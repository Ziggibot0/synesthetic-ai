"""Verify the LaTeX source for arXiv submission readiness."""
import re

with open("main.tex", encoding="utf-8") as f:
    tex = f.read()

checks = []

# 1. documentclass
checks.append(("documentclass", "\\documentclass" in tex))

# 2. begin/end document
checks.append(("begin document", "\\begin{document}" in tex))
checks.append(("end document", "\\end{document}" in tex))

# 3. title and author
checks.append(("title", "\\title{" in tex))
checks.append(("author", "\\author{" in tex))

# 4. abstract
checks.append(("abstract", "\\begin{abstract}" in tex and "\\end{abstract}" in tex))

# 5. sections
sections = re.findall(r"\\section\{([^}]+)\}", tex)
checks.append(("sections found", len(sections) >= 5))
print(f"Sections: {sections}")

# 6. tables
tables = tex.count("\\begin{table}")
checks.append(("tables", tables >= 3))
print(f"Tables: {tables}")

# 7. bibliography
checks.append(("thebibliography", "\\begin{thebibliography}" in tex))
bibitems = re.findall(r"\\bibitem\{([^}]+)\}", tex)
checks.append(("bibitems", len(bibitems) >= 6))
print(f"Bibitems: {bibitems}")

# 8. all citations resolved
cites = re.findall(r"\\cite\{([^}]+)\}", tex)
cite_keys = set()
for c in cites:
    for k in c.split(","):
        cite_keys.add(k.strip())
bib_keys = set(bibitems)
unresolved = cite_keys - bib_keys
checks.append(("all cites resolved", len(unresolved) == 0))
print(f"Citations used: {cite_keys}")
print(f"Unresolved: {unresolved}")

# 9. no unused bibitems
unused = bib_keys - cite_keys
checks.append(("no unused bibitems", len(unused) == 0))
print(f"Unused bibitems: {unused}")

# 10. balanced braces
open_b = tex.count("{")
close_b = tex.count("}")
checks.append(("balanced braces", open_b == close_b))
print(f"Braces: {open_b} open, {close_b} close, diff={open_b - close_b}")

# 11. no comment lines
comment_lines = [l for l in tex.split("\n") if l.strip().startswith("%")]
checks.append(("no comments", len(comment_lines) == 0))
print(f"Comment lines: {len(comment_lines)}")

# 12. no subdirectory references in includes
includes = re.findall(r"\\include\{([^}]+)\}", tex)
checks.append(("no subdirectory includes", all("/" not in i for i in includes)))

# 13. standard packages only
packages = re.findall(r"\\usepackage(?:\[[^\]]*\])?\{([^}]+)\}", tex)
standard = {"inputenc", "fontenc", "amsmath", "amssymb", "booktabs", "hyperref", "geometry"}
non_standard = [p for pkg in packages for p in pkg.split(",") if p.strip() not in standard]
checks.append(("standard packages only", len(non_standard) == 0))
print(f"Packages: {packages}")
print(f"Non-standard: {non_standard}")

# 14. has maketitle
checks.append(("maketitle", "\\maketitle" in tex))

# 15. word count (rough)
words = len(re.findall(r"\b[a-zA-Z]+\b", tex))
checks.append(("reasonable length", words > 1000))
print(f"Word count: {words}")

print()
print("=== VERIFICATION ===")
all_pass = True
for name, passed in checks:
    status = "PASS" if passed else "FAIL"
    if not passed:
        all_pass = False
    print(f"  {name}: {status}")
print()
print(f"OVERALL: {'ALL CHECKS PASS' if all_pass else 'SOME CHECKS FAILED'}")