#!/usr/bin/env bash
# Exported-symbol check for the consumer C ABI ([T-15], [V-11]).
#
#   ./scripts/abi/check_symbols.sh <build-dir> [<symbols-file>]
#
# Compares the neuriplo_* symbols libneuriplo exports with the committed list
# (default scripts/abi/neuriplo_c.symbols; '#' lines and blank lines ignored).
# Both directions of drift fail: a listed symbol the library does not export,
# and an exported neuriplo_* symbol that is not listed.
#
# Windows is not automated here: inspect `dumpbin /exports bin/neuriplo.dll`.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: $0 <build-dir> [<symbols-file>]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$1"
SYMBOLS_FILE="${2:-${SCRIPT_DIR}/neuriplo_c.symbols}"
export LC_ALL=C

case "$(uname -s)" in
Darwin)
    LIB="${BUILD_DIR}/libneuriplo.dylib"
    [[ -f "${LIB}" ]] || { echo "not found: ${LIB}" >&2; exit 2; }
    # Mach-O prefixes C symbols with an underscore.
    exported="$(nm -gU "${LIB}" | awk '{print $NF}' | sed 's/^_//' | grep '^neuriplo_' | sort -u || true)"
    ;;
*)
    LIB="${BUILD_DIR}/libneuriplo.so"
    [[ -f "${LIB}" ]] || { echo "not found: ${LIB}" >&2; exit 2; }
    exported="$(nm -D --defined-only "${LIB}" | awk '{print $NF}' | grep '^neuriplo_' | sort -u || true)"
    ;;
esac

listed="$(grep -v '^[[:space:]]*#' "${SYMBOLS_FILE}" | sed 's/[[:space:]]//g' | grep -v '^$' | sort -u || true)"

missing="$(comm -23 <(printf '%s\n' "${listed}") <(printf '%s\n' "${exported}") | grep -v '^$' || true)"
unlisted="$(comm -13 <(printf '%s\n' "${listed}") <(printf '%s\n' "${exported}") | grep -v '^$' || true)"

status=0
if [[ -n "${missing}" ]]; then
    echo "symbols: listed but not exported by ${LIB}:"
    printf '  %s\n' ${missing}
    status=1
fi
if [[ -n "${unlisted}" ]]; then
    echo "symbols: exported by ${LIB} but not listed in ${SYMBOLS_FILE}:"
    printf '  %s\n' ${unlisted}
    status=1
fi
if [[ ${status} -eq 0 ]]; then
    echo "symbols: OK ($(printf '%s\n' "${listed}" | grep -c .))"
fi
exit ${status}
