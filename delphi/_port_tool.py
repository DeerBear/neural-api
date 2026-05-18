#!/usr/bin/env python3
"""
FPC -> standalone-Delphi port transformer for neural-api.

Encodes the locked, reviewed recipe ONCE so it is applied consistently
across every unit (risk reduction vs. hand-transcribing ~50k lines).

What it does (and ONLY this -- everything else passes through verbatim):

1. Conditional resolution with a fixed symbol table:
     FALSE: FPC, AVX, AVX2, AVX512, AVX32, AVX64, OpenCL
     OPAQUE (kept literally, both branches preserved): everything else,
       incl. AVXANY, Release, Debug, CPUX86, CPU32, CPU64, ...
   Resolved {$IFDEF/IFNDEF X}/{$ELSE}/{$ENDIF} (block AND inline) for
   FALSE symbols are evaluated and the dead branch + the directives
   themselves are removed. Opaque directives are emitted unchanged so
   the few semantically-subtle AVXANY sites are resolved by hand later.

2. Compound-assignment expansion:  a += b;  ->  a := a + b;
   (also -=, *=, /=).  Conservative LHS; never matches := <= >= <>.
   Skips comment lines. Every rewrite is logged for audit.

It does NOT touch uses/header/type-dedup -- those are done as explicit,
greppable post-edits so they show up in review.
"""
import re, sys

FALSE_SYMS = {"FPC", "AVX", "AVX2", "AVX512", "AVX32", "AVX64", "OPENCL"}

DIRECTIVE_RE = re.compile(r"\{\$[^}]*\}")
IFDEF_RE   = re.compile(r"\{\$IF(N?)DEF\s+([A-Za-z0-9_]+)\s*\}", re.I)
IF_RE      = re.compile(r"\{\$IF[ (]", re.I)        # {$IF expr} / {$IFOPT ...}
ELSE_RE    = re.compile(r"\{\$ELSE\s*\}", re.I)
ENDIF_RE   = re.compile(r"\{\$ENDIF[^}]*\}", re.I)

def classify(d):
    m = IFDEF_RE.match(d)
    if m:
        neg = m.group(1).upper() == "N"
        sym = m.group(2).upper()
        if sym in FALSE_SYMS:
            # IFDEF false-sym -> branch inactive; IFNDEF false-sym -> active
            return ("resolve_if", (True if neg else False))
        return ("opaque_if", None)
    if IF_RE.match(d):              # {$IF ...}/{$IFOPT ...} -> opaque
        return ("opaque_if", None)
    if ELSE_RE.match(d):
        return ("else", None)
    if ENDIF_RE.match(d):
        return ("endif", None)
    return ("other", None)         # plain directive ({$R-}, {$include}, ...)

def transform_conditionals(text):
    # Split into [text, directive, text, directive, ...]
    parts = DIRECTIVE_RE.split(text)
    dirs  = DIRECTIVE_RE.findall(text)
    out = []
    # stack frames: dict(kind='resolve'|'opaque', emit=bool, taken=bool)
    stack = []
    def emitting():
        for f in stack:
            if f["kind"] == "resolve" and not f["emit"]:
                return False
        return True
    for i, seg in enumerate(parts):
        if emitting():
            out.append(seg)
        if i < len(dirs):
            d = dirs[i]
            kind, val = classify(d)
            if kind == "resolve_if":
                parent = emitting()
                active = parent and val
                stack.append({"kind": "resolve", "emit": active,
                              "taken": active, "parent": parent})
            elif kind == "opaque_if":
                if emitting():
                    out.append(d)
                stack.append({"kind": "opaque", "emit": True,
                              "taken": True, "parent": emitting()})
            elif kind == "else":
                if not stack:
                    out.append(d); continue
                f = stack[-1]
                if f["kind"] == "resolve":
                    if f["parent"] and not f["taken"]:
                        f["emit"] = True; f["taken"] = True
                    else:
                        f["emit"] = False
                else:
                    if emitting(): out.append(d)
            elif kind == "endif":
                if not stack:
                    out.append(d); continue
                f = stack.pop()
                if f["kind"] == "opaque" and emitting():
                    out.append(d)
            else:  # plain directive: emit if active
                if emitting():
                    out.append(d)
    return "".join(out)

# --- compound assignment expansion -----------------------------------------
CMPD_RE = re.compile(
    r'^(?P<ind>\s*)(?P<lhs>[A-Za-z_][^=;]*?)\s*(?<![:<>=!])(?P<op>[-+*/])=\s*'
    r'(?P<rhs>[^;]+);(?P<tail>[ \t]*(?://.*)?)$')

def expand_compound(text, log):
    res = []
    for ln, line in enumerate(text.split("\n"), 1):
        s = line.lstrip()
        if s.startswith("//") or s.startswith("(*") or s.startswith("{"):
            res.append(line); continue
        m = CMPD_RE.match(line)
        if m and line.count("'", 0, m.start("op")) % 2 == 1:
            m = None   # operator sits inside a string literal -> not a compound assign
        if m:
            lhs = m["lhs"].rstrip()
            new = f'{m["ind"]}{lhs} := {lhs} {m["op"]} {m["rhs"].strip()};{m["tail"]}'
            log.append((ln, line.strip(), new.strip()))
            res.append(new)
        else:
            res.append(line)
    return "\n".join(res)

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    raw = open(src, encoding="utf-8", errors="replace").read()
    step1 = transform_conditionals(raw)
    log = []
    step2 = expand_compound(step1, log)
    open(dst, "w", encoding="utf-8").write(step2)
    sys.stderr.write(f"compound-assignment rewrites: {len(log)}\n")
    for ln, a, b in log:
        sys.stderr.write(f"  L{ln}: {a}   ->   {b}\n")
