#!/usr/bin/env python3
"""
FPC -> standalone-Delphi port transformer for neural-api.

Encodes the locked, reviewed recipe ONCE so it is applied consistently
across every unit (risk reduction vs. hand-transcribing ~50k lines).

1. Conditional resolution with a fixed symbol table:
     FALSE: FPC, AVX, AVX2, AVX512, AVX32, AVX64, OpenCL
     OPAQUE (kept literally, both branches preserved): everything else,
       incl. AVXANY, Release, Debug, CPUX86, CPU32, CPU64, ...
   Resolved {$IFDEF/IFNDEF X}/{$ELSE}/{$ENDIF} (block AND inline) for
   FALSE symbols are evaluated and the dead branch + the directives
   themselves are removed. Opaque directives are emitted unchanged so
   the few semantically-subtle AVXANY sites are resolved by hand later.

2. Compound-assignment expansion:  a += b;  ->  a := a + b;  (-=,*=,/=)

CRITICAL: directive detection is Pascal-lexer-aware. A `{$...}` token is
only a compiler directive when it occurs in normal code -- NEVER inside
a string literal, a // line comment, a { } brace comment, or a (* *)
block comment. neural-api keeps large (* *)-commented code blocks that
contain {$IFDEF ...}; treating those as live would desync resolution.
The lexer also tells the compound expander which lines begin inside a
comment so it never rewrites commented-out code.
"""
import re, sys

FALSE_SYMS = {"FPC", "AVX", "AVX2", "AVX512", "AVX32", "AVX64", "OPENCL"}

# --- Pascal-aware lexer ----------------------------------------------------
# Yields ("code", s) | ("dir", s) | ("cmt", s).  "code" is live source
# (directive resolution + compound expansion may act on it); "dir" is a
# genuine {$...} directive; "cmt" is string/comment text passed through
# verbatim and never interpreted.  Also returns, per output line, whether
# that line STARTS inside a (* *) or { } comment.

def lex(text):
    toks = []
    i, n = 0, len(text)
    buf = []
    def flush_code():
        if buf:
            toks.append(("code", "".join(buf)))
            buf.clear()
    while i < n:
        c = text[i]
        c2 = text[i:i+2]
        if c == "'":                                  # string literal
            flush_code()
            j = i + 1
            while j < n:
                if text[j] == "'":
                    if text[j:j+2] == "''":
                        j += 2; continue
                    j += 1; break
                if text[j] == "\n":
                    break
                j += 1
            toks.append(("cmt", text[i:j])); i = j; continue
        if c2 == "//":                                # line comment
            flush_code()
            j = text.find("\n", i)
            if j == -1: j = n
            toks.append(("cmt", text[i:j])); i = j; continue
        if c2 == "(*":                                # paren block comment
            flush_code()
            j = text.find("*)", i + 2)
            j = n if j == -1 else j + 2
            toks.append(("cmt", text[i:j])); i = j; continue
        if c == "{":
            if i + 1 < n and text[i+1] == "$":         # compiler directive
                flush_code()
                j = text.find("}", i)
                j = n if j == -1 else j + 1
                toks.append(("dir", text[i:j])); i = j; continue
            else:                                      # brace comment
                flush_code()
                j = text.find("}", i)
                j = n if j == -1 else j + 1
                toks.append(("cmt", text[i:j])); i = j; continue
        buf.append(c); i += 1
    flush_code()
    return toks

IFDEF_RE = re.compile(r"\{\$IF(N?)DEF\s+([A-Za-z0-9_]+)\s*\}", re.I)
IF_RE    = re.compile(r"\{\$IF[ (]", re.I)
ELSE_RE  = re.compile(r"\{\$ELSE\s*\}", re.I)
ENDIF_RE = re.compile(r"\{\$ENDIF[^}]*\}", re.I)

def classify(d):
    m = IFDEF_RE.match(d)
    if m:
        neg = m.group(1).upper() == "N"
        sym = m.group(2).upper()
        if sym in FALSE_SYMS:
            return ("resolve_if", True if neg else False)
        return ("opaque_if", None)
    if IF_RE.match(d):
        return ("opaque_if", None)
    if ELSE_RE.match(d):
        return ("else", None)
    if ENDIF_RE.match(d):
        return ("endif", None)
    return ("other", None)

def transform_conditionals(text):
    toks = lex(text)
    out = []
    stack = []
    def emitting():
        return all(not (f["kind"] == "resolve" and not f["emit"]) for f in stack)
    for kind, s in toks:
        if kind in ("code", "cmt"):
            if emitting():
                out.append(s)
            continue
        d = s
        k, val = classify(d)
        if k == "resolve_if":
            parent = emitting()
            active = parent and val
            stack.append({"kind": "resolve", "emit": active,
                          "taken": active, "parent": parent})
        elif k == "opaque_if":
            if emitting(): out.append(d)
            stack.append({"kind": "opaque", "emit": True,
                          "taken": True, "parent": emitting()})
        elif k == "else":
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
        elif k == "endif":
            if not stack:
                out.append(d); continue
            f = stack.pop()
            if f["kind"] == "opaque" and emitting():
                out.append(d)
        else:
            if emitting():
                out.append(d)
    return "".join(out)

# --- compound assignment expansion -----------------------------------------
CMPD_RE = re.compile(
    r'^(?P<ind>\s*)(?P<lhs>[A-Za-z_][^=;]*?)\s*(?<![:<>=!])(?P<op>[-+*/])=\s*'
    r'(?P<rhs>[^;]+);(?P<tail>[ \t]*(?://.*)?)$')

def comment_open_at_line_start(text):
    """Return set of 1-based line numbers whose start is inside (* *) or { }."""
    inside = set()
    state = 0   # 0 normal, 1 paren-comment, 2 brace-comment
    line = 1
    i, n = 0, len(text)
    while i < n:
        c = text[i]; c2 = text[i:i+2]
        if c == "\n":
            line += 1
            if state in (1, 2):
                inside.add(line)
            i += 1; continue
        if state == 1:
            if c2 == "*)": state = 0; i += 2; continue
            i += 1; continue
        if state == 2:
            if c == "}": state = 0
            i += 1; continue
        if c == "'":
            j = i + 1
            while j < n and text[j] != "\n":
                if text[j] == "'":
                    if text[j:j+2] == "''": j += 2; continue
                    j += 1; break
                j += 1
            i = j; continue
        if c2 == "//":
            j = text.find("\n", i); i = n if j == -1 else j; continue
        if c2 == "(*":
            state = 1; i += 2; continue
        if c == "{":
            if i + 1 < n and text[i+1] == "$":
                j = text.find("}", i); i = n if j == -1 else j + 1; continue
            state = 2; i += 1; continue
        i += 1
    return inside

def expand_compound(text, log):
    skip = comment_open_at_line_start(text)
    res = []
    for ln, line in enumerate(text.split("\n"), 1):
        s = line.lstrip()
        if ln in skip or s.startswith("//") or s.startswith("(*") or s.startswith("{"):
            res.append(line); continue
        m = CMPD_RE.match(line)
        if m and line.count("'", 0, m.start("op")) % 2 == 1:
            m = None
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
