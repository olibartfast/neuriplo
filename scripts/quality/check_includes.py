#!/usr/bin/env python3
"""Flag standard-library symbols used without the header that declares them.

Why this exists: the tree used to reach several std headers only because
<opencv2/...> pulled them in. Removing OpenCV from the backends that never
used it turned a handful of files that had always been missing an include
into compile errors -- but only on the toolchains that did not paper over
them. A missing <cassert> is invisible under libstdc++ and fatal under MSVC,
so the breakage surfaced late, in a 30-minute Docker matrix, one backend at
a time.

This finds the same class of bug in seconds, with no compiler and no SDKs.

Method: for each translation unit, resolve the closure of project headers it
includes, collect every <system header> named anywhere in that closure, then
check each std symbol the file actually uses against the header that
provides it. Crediting the whole closure is deliberate -- a symbol reached
through the project's own headers is a normal, intended dependency. What it
will not credit is one std header dragging in another, which is exactly the
guarantee no standard makes and no two toolchains agree on.

Exit status is 1 if anything is reported, 0 otherwise.
"""

import argparse
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Directories that are never ours to fix.
SKIP_DIRS = {".git", "build", "cmake-build-debug", "cmake-build-release", "third_party", "external", "vcpkg", "_deps"}

SOURCE_SUFFIXES = (".cpp", ".cc", ".cxx", ".hpp", ".hh", ".h")

# Where a quoted include may resolve, relative to the repo root, on top of the
# including file's own directory. Mirrors the target_include_directories the
# CMake files set up -- every backend's src/ is on the include path of its own
# tests, so `#include "CactusInfer.hpp"` from backends/cactus/test resolves
# into backends/cactus/src.
#
# Getting this list wrong does not fail loudly on its own: an unresolvable
# project header silently truncates the closure, every std header behind it
# goes missing, and the tool reports a pile of includes that are in fact
# present. So an include that resolves to nothing is itself an error.
INCLUDE_ROOTS = [
    "include",
    "backends/src",
    "backends/src/plugin",
    "src",
] + sorted(
    os.path.relpath(os.path.join(REPO_ROOT, "backends", entry, "src"), REPO_ROOT)
    for entry in os.listdir(os.path.join(REPO_ROOT, "backends"))
    if os.path.isdir(os.path.join(REPO_ROOT, "backends", entry, "src"))
)

# symbol -> header that declares it. Kept to symbols whose home header is
# unambiguous; std::abs and std::max live in more than one and are left out
# rather than guessed at.
SYMBOL_HEADERS = {
    "cassert": ["assert"],
    "cmath": [
        "std::isfinite", "std::isnan", "std::isinf", "std::sqrt", "std::pow",
        "std::floor", "std::ceil", "std::round", "std::fabs", "std::exp", "std::log",
    ],
    "cstdlib": ["std::getenv", "std::exit", "std::abort", "std::strtol", "std::strtod", "std::atoi"],
    "cstring": ["std::memcpy", "std::memset", "std::memcmp", "std::strlen", "std::strcmp", "std::strncmp"],
    "cstdint": ["int8_t", "int16_t", "int32_t", "int64_t", "uint8_t", "uint16_t", "uint32_t", "uint64_t"],
    "string": ["std::string", "std::to_string", "std::stoi", "std::stof", "std::stod"],
    "string_view": ["std::string_view"],
    "vector": ["std::vector"],
    "array": ["std::array"],
    "map": ["std::map"],
    "unordered_map": ["std::unordered_map"],
    "set": ["std::set"],
    "unordered_set": ["std::unordered_set"],
    "memory": ["std::unique_ptr", "std::shared_ptr", "std::make_unique", "std::make_shared", "std::weak_ptr"],
    "algorithm": [
        "std::sort", "std::find", "std::find_if", "std::find_if_not", "std::transform",
        "std::any_of", "std::all_of", "std::none_of", "std::copy", "std::fill", "std::count_if",
    ],
    "numeric": ["std::accumulate", "std::iota", "std::inner_product"],
    "sstream": ["std::ostringstream", "std::istringstream", "std::stringstream"],
    "fstream": ["std::ifstream", "std::ofstream", "std::fstream"],
    "iostream": ["std::cout", "std::cerr", "std::cin"],
    "ostream": ["std::endl"],
    "iomanip": ["std::setprecision", "std::setw", "std::setfill"],
    "stdexcept": [
        "std::runtime_error", "std::logic_error", "std::invalid_argument",
        "std::out_of_range", "std::length_error", "std::domain_error",
    ],
    "mutex": ["std::mutex", "std::lock_guard", "std::unique_lock", "std::scoped_lock"],
    "thread": ["std::thread"],
    "chrono": ["std::chrono"],
    "functional": ["std::function", "std::bind", "std::ref"],
    "optional": ["std::optional", "std::nullopt"],
    "variant": ["std::variant", "std::holds_alternative", "std::visit"],
    "tuple": ["std::tuple", "std::make_tuple", "std::tie"],
    "utility": ["std::move", "std::forward", "std::pair", "std::make_pair"],
    "limits": ["std::numeric_limits"],
    "type_traits": ["std::is_same", "std::enable_if", "std::decay", "std::remove_reference"],
    "filesystem": ["std::filesystem"],
    "cstddef": ["std::size_t", "std::ptrdiff_t"],
}

# provider -> the requirements that having it satisfies. The direction matters:
# this is read as "the file includes KEY, so a need for any of VALUE is met",
# never the reverse. Kept to the C/C++ spellings of one another, which are the
# same header, plus <sstream>, which cannot define basic_stringstream without
# basic_string. Notably absent: <cstddef> does not satisfy <cstdint> (it
# declares size_t, not int64_t), and <iostream> does not satisfy <string>.
PROVIDES = {
    "cstdint": {"stdint.h"},
    "stdint.h": {"cstdint"},
    "cstddef": {"stddef.h"},
    "stddef.h": {"cstddef"},
    "cstring": {"string.h"},
    "string.h": {"cstring"},
    "cstdlib": {"stdlib.h"},
    "stdlib.h": {"cstdlib"},
    "cmath": {"math.h"},
    "math.h": {"cmath"},
    "cassert": {"assert.h"},
    "assert.h": {"cassert"},
    "sstream": {"string", "ostream"},
    # [iostream.syn] says <iostream> includes <ostream>, so std::endl is
    # available through it. <fstream> and <sstream> likewise cannot declare
    # basic_ofstream/basic_ostringstream without basic_ostream.
    "iostream": {"ostream"},
    "fstream": {"ostream"},
}

# Quoted includes that belong to an SDK rather than to this repo. Without this,
# an SDK that spells its own includes with quotes reads as a hole in the
# closure. Adding to this list is the intended fix when the check reports one.
EXTERNAL_QUOTED_PREFIXES = ("openvino/",)

CONTAINER_HEADERS = {"array", "vector", "map", "unordered_map", "set", "unordered_set", "string", "tuple", "utility"}

# Unqualified names that are function-like; everything else spelled without a
# std:: prefix in SYMBOL_HEADERS is a type name.
FUNCTION_LIKE_BARE = {"assert"}

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*(?:"([^"]+)"|<([^>]+)>)', re.MULTILINE)


def strip_comments_and_strings(text):
    """Blank out comments and literals so matches come from real code only."""
    out = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            # keep newlines so line numbers survive
            out.append("".join(ch if ch == "\n" else " " for ch in text[i:j]))
            i = j
        elif c in ('"', "'"):
            quote = c
            j = i + 1
            while j < n:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == quote or text[j] == "\n":
                    j += 1
                    break
                j += 1
            out.append("".join(ch if ch == "\n" else " " for ch in text[i:j]))
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def resolve_quoted(include, from_file):
    candidates = [os.path.join(os.path.dirname(from_file), include)]
    candidates += [os.path.join(REPO_ROOT, root, include) for root in INCLUDE_ROOTS]
    # Backend sources include their neighbours by bare name.
    candidates.append(os.path.join(REPO_ROOT, "backends", os.path.basename(include)))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.normpath(candidate)
    return None


def closure(path, cache, seen=None, unresolved=None):
    """Every system header named in `path` or in the project headers it pulls in.

    Quoted includes that resolve to no file on disk are recorded in
    `unresolved` rather than ignored: each one is a hole in the closure that
    would otherwise turn into spurious findings.
    """
    if seen is None:
        seen = set()
    if path in seen:
        return set()
    seen.add(path)
    if path in cache:
        system, quoted = cache[path]
    else:
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                text = handle.read()
        except OSError:
            return set()
        system, quoted = set(), []
        for match in INCLUDE_RE.finditer(text):
            if match.group(1) is not None:
                quoted.append(match.group(1))
            else:
                system.add(match.group(2))
        cache[path] = (system, quoted)

    headers = set(system)
    for include in quoted:
        if include.startswith(EXTERNAL_QUOTED_PREFIXES):
            continue
        resolved = resolve_quoted(include, path)
        if resolved is None:
            if unresolved is not None:
                unresolved.add((os.path.relpath(path, REPO_ROOT), include))
        else:
            headers |= closure(resolved, cache, seen, unresolved)
    return headers


def satisfied(required, available):
    if required in available:
        return True
    if any(required in PROVIDES.get(header, ()) for header in available):
        return True
    # A size_t requirement is met by any container header.
    if required == "cstddef" and (available & CONTAINER_HEADERS):
        return True
    return False


def symbol_pattern(symbol):
    """Only the qualified spelling, deliberately.

    An unqualified fallback of the form `\\bname\\s*\\(` cannot tell a call from
    a declaration without parsing: `void log(Severity, const char*)` looks
    exactly like a call to std::log. Matching `std::name` costs some recall on
    files with `using namespace std;` -- which this tree does not have -- and
    buys a report with no false positives, which is what a blocking check
    needs. `assert` stays unqualified because it is a macro and never a member.
    """
    if symbol.startswith("std::"):
        return re.compile(r"\bstd\s*::\s*" + re.escape(symbol[5:]) + r"\b")
    if symbol in FUNCTION_LIKE_BARE:
        return re.compile(r"(?<![\w:.])" + re.escape(symbol) + r"\s*\(")
    # A bare type name (int64_t, uint8_t): an identifier, not a call.
    return re.compile(r"(?<![\w:.])" + re.escape(symbol) + r"\b")


PATTERNS = {
    header: [(symbol, symbol_pattern(symbol)) for symbol in symbols]
    for header, symbols in SYMBOL_HEADERS.items()
    if symbols
}


def iter_sources(roots):
    for root in roots:
        base = os.path.join(REPO_ROOT, root)
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
            for name in sorted(filenames):
                if name.endswith(SOURCE_SUFFIXES):
                    yield os.path.join(dirpath, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="*", default=["include", "src", "backends", "test"],
                        help="directories to scan, relative to the repo root")
    args = parser.parse_args()

    cache = {}
    findings = []
    unresolved = set()

    for path in iter_sources(args.roots):
        try:
            with open(path, encoding="utf-8", errors="replace") as handle:
                text = handle.read()
        except OSError:
            continue
        code = strip_comments_and_strings(text)
        available = closure(path, cache, unresolved=unresolved)

        for header, patterns in PATTERNS.items():
            if satisfied(header, available):
                continue
            for symbol, pattern in patterns:
                match = pattern.search(code)
                if match is None:
                    continue
                line = code.count("\n", 0, match.start()) + 1
                findings.append((os.path.relpath(path, REPO_ROOT), line, symbol, header))
                break  # one finding per missing header per file is enough

    for path, line, symbol, header in sorted(findings):
        print(f"{path}:{line}: {symbol} used without <{header}>")

    if unresolved:
        print("\nUnresolved project includes -- INCLUDE_ROOTS needs updating; findings "
              "above may be wrong until it is:", file=sys.stderr)
        for path, include in sorted(unresolved):
            print(f"  {path}: \"{include}\"", file=sys.stderr)
        return 1

    if findings:
        print(f"\n{len(findings)} missing include(s). "
              f"Add the header to the file that uses the symbol.", file=sys.stderr)
        return 1
    print(f"No missing includes across {len(cache)} file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
