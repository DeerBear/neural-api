#!/usr/bin/env python3
"""
FPC/Lazarus example -> standalone-Delphi .dpr generator for neural-api.

Companion to _port_tool.py (which ports the *library* units under neural/).
This script ports the *example programs* under examples/ into self-contained
Delphi console projects under delphi/examples/, mirroring the source layout.

Per console example program (.lpr or .pas) it:
  * resolves {$IFDEF FPC/AVX*/OpenCL/UNIX/UseCThreads} as false -- this drops
    the `cthreads` uses-fragment and FPC-only branches -- reusing _port_tool's
    reviewed, comment-aware conditional resolver;
  * removes {$mode ...}/{$H+} directives and inserts {$APPTYPE CONSOLE};
  * expands compound assignments (+=, -=, *=, /=);
  * rewrites the uses clause so every unit ported into delphi/neural/, the
    CustApp shim, and any companion unit carries an explicit relative path
    (Name in '..\\..\\neural\\Name.pas') -- making each .dpr self-contained,
    with no search-path or .dproj setup needed.

Companion units (non-program .pas in the same source dir, referenced from the
program's uses clause) are ported the same way -- minus the program-only
steps -- and copied next to the .dpr.

No .dproj is generated: Delphi creates the project file itself on first open.
The 4 LCL-GUI examples are out of scope and skipped.
"""
import os, re, sys, shutil
import _port_tool as pt

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES_SRC = os.path.join(REPO, "examples")
DELPHI_DIR = os.path.join(REPO, "delphi")
NEURAL_DIR = os.path.join(DELPHI_DIR, "neural")
DEST_ROOT = os.path.join(DELPHI_DIR, "examples")

# LCL-GUI examples (have .lfm form files) -- out of scope, handled separately.
GUI_PROJECTS = {"GradientAscent", "SuperResolutionApp", "VisualAutoencoder", "VisualGAN"}

# The Delphi examples target Windows, so UNIX / UseCThreads branches are dead
# code. Extend the reviewed resolver's false-symbol set for the example context.
pt.FALSE_SYMS = pt.FALSE_SYMS | {"UNIX", "USECTHREADS"}

# lowercased unit name -> actual filename, for every ported library unit.
NEURAL_UNITS = {
    os.path.splitext(f)[0].lower(): f
    for f in os.listdir(NEURAL_DIR) if f.lower().endswith(".pas")
}

MODE_RE = re.compile(r'\{\$(?:mode|modeswitch|h[+-])[^}]*\}', re.I)
PROGRAM_RE = re.compile(r'\bprogram\s+[A-Za-z_]\w*\s*;', re.I)
USES_KW = re.compile(r'\buses\b', re.I)


def read(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def list_console_projects():
    """Return [(lpi_path, program_path), ...] for every non-GUI example."""
    found = []
    for dirpath, _, files in os.walk(EXAMPLES_SRC):
        for f in sorted(files):
            if not f.endswith(".lpi"):
                continue
            base = f[:-4]
            if base in GUI_PROJECTS:
                continue
            stem = os.path.join(dirpath, base)
            if os.path.isfile(stem + ".lpr"):
                prog = stem + ".lpr"
            elif os.path.isfile(stem + ".pas"):
                prog = stem + ".pas"
            else:
                sys.stderr.write("WARN: no program file for %s\n" % f)
                continue
            found.append((os.path.join(dirpath, f), prog))
    return sorted(found)


def split_uses(body):
    """Unit identifiers, in order, from a (comment-free) uses-clause body."""
    names = []
    for part in body.split(","):
        p = part.strip()
        if p:
            names.append(re.split(r"\s+", p)[0])
    return names


def find_program_uses(text):
    """Locate the program's uses clause via _port_tool's Pascal-aware lexer.

    Returns (start, end, [unit names]) spanning the whole `uses ... ;`
    clause, or None. Comments and directives inside the clause are ignored,
    so a `//` line comment or a `;` inside a comment never derails parsing.
    """
    pos, spans = 0, []
    for kind, s in pt.lex(text):
        spans.append((pos, kind, s))
        pos += len(s)
    start = start_tok = None
    for i, (a, kind, s) in enumerate(spans):
        if kind == "code":
            m = USES_KW.search(s)
            if m:
                start, start_tok = a + m.start(), i
                break
    if start is None:
        return None
    body, end = [], None
    for i in range(start_tok, len(spans)):
        a, kind, s = spans[i]
        if kind != "code":
            continue
        seg, seg_off = s, a
        if i == start_tok:
            kwend = USES_KW.search(s).end()
            seg, seg_off = s[kwend:], a + kwend
        semi = seg.find(";")
        if semi != -1:
            body.append(seg[:semi])
            end = seg_off + semi + 1
            break
        body.append(seg)
    if end is None:
        return None
    return start, end, split_uses("".join(body))


def fpc_cleanup(text):
    """Conditional resolution + mode-directive removal + compound expansion."""
    text = pt.transform_conditionals(text)
    text = MODE_RE.sub("", text)
    text = pt.expand_compound(text, [])
    return text


# --- neural-unit dependency closure ----------------------------------------
# Delphi compiles each unit a .dpr references with `in '...'`, but does not
# reliably resolve that unit's *own* dependencies from a sibling path. So the
# .dpr must list the full transitive closure of neural units explicitly.

ALL_USES_RE = re.compile(r"\buses\b(.*?);", re.I | re.S)


def code_only(text):
    """Just the live code: string/comment AND directive tokens removed."""
    return "".join(s for kind, s in pt.lex(text) if kind == "code")


def neural_uses(text):
    """Lowercased neural-unit names referenced by any uses clause in text."""
    deps = set()
    for m in ALL_USES_RE.finditer(code_only(text)):
        for name in split_uses(m.group(1)):
            if name.lower() in NEURAL_UNITS:
                deps.add(name.lower())
    return deps


def build_neural_graph():
    """Map each delphi/neural unit -> the neural units it directly uses."""
    graph = {}
    for low, fname in NEURAL_UNITS.items():
        graph[low] = neural_uses(read(os.path.join(NEURAL_DIR, fname))) - {low}
    return graph


NEURAL_GRAPH = build_neural_graph()


def neural_closure(seeds):
    """Transitive closure of neural-unit dependencies over NEURAL_GRAPH."""
    seen, stack = set(), list(seeds)
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        stack.extend(NEURAL_GRAPH.get(u, set()) - seen)
    return seen


def add_apptype(text):
    if re.search(r"\{\$APPTYPE\b", text, re.I):
        return text
    m = PROGRAM_RE.search(text)
    if not m:
        return text
    return text[:m.end()] + "\n\n{$APPTYPE CONSOLE}" + text[m.end():]


def rewrite_uses(text, depth, companions, extra_neural):
    """Add explicit `in '...'` paths for neural/shim/companion units.

    depth: directory levels of the .dpr's dir below delphi/examples/
           (Hypotenuse -> 1, ResNet/server -> 2).
    companions: set of lowercased companion unit names.
    extra_neural: sorted lowercased neural units to append (closure members
           not already named in the original uses clause).
    """
    found = find_program_uses(text)
    if not found:
        return text
    start, end, names = found
    neural_rel = "..\\" * (depth + 1) + "neural\\"
    custapp_rel = "..\\" * depth

    def neural_entry(low):
        fname = NEURAL_UNITS[low]
        return "%s in '%s%s'" % (os.path.splitext(fname)[0], neural_rel, fname)

    out = []
    for name in names:
        low = name.lower()
        if low in NEURAL_UNITS:
            out.append("%s in '%s%s'" % (name, neural_rel, NEURAL_UNITS[low]))
        elif low == "custapp":
            out.append("CustApp in '%sCustApp.pas'" % custapp_rel)
        elif low in companions:
            out.append("%s in '%s.pas'" % (name, name))
        else:
            out.append(name)
    out.extend(neural_entry(low) for low in extra_neural)
    new_uses = "uses\n  " + ",\n  ".join(out) + ";"
    return text[:start] + new_uses + text[end:]


def process_project(prog_path):
    src_dir = os.path.dirname(prog_path)
    rel_dir = os.path.relpath(src_dir, EXAMPLES_SRC).replace("\\", "/")
    depth = len(rel_dir.split("/"))
    dest_dir = os.path.join(DEST_ROOT, *rel_dir.split("/"))
    prog_name = os.path.splitext(os.path.basename(prog_path))[0]

    raw = read(prog_path)
    resolved = pt.transform_conditionals(raw)

    # Companion units = .pas files in the source dir named in the uses clause.
    companions = {}
    found = find_program_uses(resolved)
    if found:
        for name in found[2]:
            cand = os.path.join(src_dir, name + ".pas")
            if name.lower() not in NEURAL_UNITS and os.path.isfile(cand):
                companions[name.lower()] = (name, cand)

    # Full neural-unit closure: seed from the units the program and its
    # companions use directly, then append every transitive dependency that
    # is not already named in the program's own uses clause.
    prog_uses_neural = neural_uses(raw)
    seeds = set(prog_uses_neural)
    for _, cand in companions.values():
        seeds |= neural_uses(read(cand))
    extra = sorted(neural_closure(seeds) - prog_uses_neural)

    text = add_apptype(fpc_cleanup(raw))
    text = rewrite_uses(text, depth, set(companions), extra)
    write(os.path.join(dest_dir, prog_name + ".dpr"), text)

    for unit_name, cand in companions.values():
        write(os.path.join(dest_dir, unit_name + ".pas"),
              fpc_cleanup(read(cand)))

    return rel_dir, prog_name, sorted(n for n, _ in companions.values())


def clean_dest():
    """Remove generated project subdirs, keep CustApp.pas / README.md."""
    if not os.path.isdir(DEST_ROOT):
        return
    for name in os.listdir(DEST_ROOT):
        p = os.path.join(DEST_ROOT, name)
        if os.path.isdir(p):
            shutil.rmtree(p)


def main():
    clean_dest()
    projects = list_console_projects()
    companions_total = 0
    for _, prog_path in projects:
        rel_dir, prog_name, comps = process_project(prog_path)
        companions_total += len(comps)
        extra = ("  (+%s)" % ", ".join(comps)) if comps else ""
        sys.stdout.write("%-58s %s.dpr%s\n" % (rel_dir + "/", prog_name, extra))
    sys.stdout.write("\n%d console projects, %d companion units ported.\n"
                     % (len(projects), companions_total))


if __name__ == "__main__":
    main()
