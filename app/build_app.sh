#!/usr/bin/env bash
# Build the double-clickable Automix.app -- offline, from installed packages.
#
#   ./build_app.sh            # build into dist/Automix.app
#   open dist/Automix.app     # run it
#
# Notes that cost me an afternoon:
#  * multiprocessing.freeze_support() must be in app.py, or every render child
#    re-executes the bundle and starts a second server instead of the worker.
#  * web/ and presets/ are bundled as data, and server.py looks for them under
#    sys._MEIPASS when frozen.
#  * the bundle is ad-hoc signed by PyInstaller.  Moving it to another Mac will
#    trip Gatekeeper; right-click -> Open, or `xattr -d com.apple.quarantine`.
#
# The build never reaches the network.  It does not download and it does not
# install anything: it uses an interpreter that already has everything, and
# when something is missing it says so instead of fetching it.
#
#   AUTOMIX_BUILD_PYTHON=/path/to/python3 ./build_app.sh
#
# Keep the earlier Apps by giving this build its own output folders:
#
#   AUTOMIX_BUILD_DIST=dist-20261030-r31 \
#   AUTOMIX_BUILD_WORK=build-20261030-r31 \
#   AUTOMIX_BUILD_SPEC=build-20261030-r31 \
#   AUTOMIX_BUILD_PYTHON=/path/to/python3 ./build_app.sh
set -euo pipefail
cd "$(dirname "$0")"

# PyInstaller resolves a relative input against the folder the spec file lands
# in, not against this script: with a separate --specpath, "web" would be
# looked up next to the spec and not found.  So every source path is absolute.
SRC_ROOT="$(pwd)"
REPO_ROOT="$(dirname "$SRC_ROOT")"

BUILD_PYTHON="${AUTOMIX_BUILD_PYTHON:-python3}"
DIST_DIR="${AUTOMIX_BUILD_DIST:-dist}"
WORK_DIR="${AUTOMIX_BUILD_WORK:-build}"
SPEC_DIR="${AUTOMIX_BUILD_SPEC:-$WORK_DIR}"
BUILD_LOG="${AUTOMIX_BUILD_LOG:-/tmp/automix-build.log}"

# 1. The interpreter has to exist.  There is no fallback to another one and
#    no attempt to install anything.
if [[ "$BUILD_PYTHON" == */* ]]; then
  if [[ ! -x "$BUILD_PYTHON" ]]; then
    echo "build python is not executable: $BUILD_PYTHON" >&2
    exit 1
  fi
else
  if ! command -v "$BUILD_PYTHON" >/dev/null 2>&1; then
    echo "build python not found on PATH: $BUILD_PYTHON" >&2
    exit 1
  fi
fi
echo "build python: $BUILD_PYTHON ($("$BUILD_PYTHON" -c 'import platform;print(platform.python_version())'))"

# 2. Everything the build and the bundle need must already be installed.
#    The Python Schema validator is mandatory now: a plan is only accepted
#    when the real validator says so, so the App has to carry it.
CHECK_OUT="$("$BUILD_PYTHON" - <<'PY'
import importlib
import importlib.metadata as md
import sys

REQUIRED = {
    "jsonschema": "4.26.0",
    "jsonschema_specifications": "2025.9.1",
    "referencing": "0.37.0",
    "rpds-py": "0.30.0",
    "attrs": "26.1.0",
}
#: the distribution name is not always the module name
MODULES = {
    "jsonschema": "jsonschema",
    "jsonschema_specifications": "jsonschema_specifications",
    "referencing": "referencing",
    "rpds-py": "rpds",
    "attrs": "attrs",
}
BUILD_ONLY = ("numpy", "PyInstaller")

problems = []
lines = []
for name, wanted in REQUIRED.items():
    try:
        found = md.version(name)
    except md.PackageNotFoundError:
        problems.append(f"{name} is not installed in this interpreter")
        continue
    try:
        importlib.import_module(MODULES[name])
    except ImportError as exc:
        problems.append(f"{name} is installed but not importable: {exc}")
        continue
    lines.append(f"VERSION {name} {found}")
    if found != wanted:
        problems.append(
            f"{name} {found} is installed, the bundle is expected to carry {wanted}"
        )
    try:
        dist = md.distribution(name)
        lines.append(f"DISTINFO {name} {dist._path}")
    except Exception:
        pass

for name in BUILD_ONLY:
    try:
        module = importlib.import_module(name)
    except ImportError as exc:
        problems.append(f"{name} is not importable: {exc}")
        continue
    lines.append(f"VERSION {name} {getattr(module, '__version__', 'unknown')}")

for line in lines:
    print(line)
if problems:
    print("PROBLEMS:")
    for problem in problems:
        print(problem)
    sys.exit(1)
PY
)" || {
  echo "" >&2
  echo "the build needs packages that are not ready in $BUILD_PYTHON:" >&2
  echo "$CHECK_OUT" | sed -n '/^VERSION /s/^VERSION /  have /p' >&2
  echo "$CHECK_OUT" | sed -n '/^PROBLEMS:/,$p' | tail -n +2 | sed 's/^/  - /' >&2
  echo "" >&2
  echo "install them into that interpreter (pip install jsonschema==4.26.0 \\" >&2
  echo "    jsonschema-specifications==2025.9.1 referencing==0.37.0 rpds-py==0.30.0 \\" >&2
  echo "    attrs==26.1.0 pyinstaller numpy) or point AUTOMIX_BUILD_PYTHON at an" >&2
  echo "interpreter that already has them -- this build never downloads anything." >&2
  exit 1
}
echo "$CHECK_OUT" | sed -n 's/^VERSION /  /p'

# 3. The validator's metadata and licenses go into the bundle as data: the
#    App has to be able to say what it is running on.
ADD_DATA_ARGS=()
COPY_METADATA_ARGS=()
while read -r kind name path; do
  [[ "$kind" == "DISTINFO" ]] || continue
  COPY_METADATA_ARGS+=(--copy-metadata "$name")
  while IFS= read -r license; do
    ADD_DATA_ARGS+=(--add-data "$license:$name-licenses")
  done < <(find "$path" -maxdepth 2 -type f \( -iname 'LICENSE*' -o -iname 'COPYING*' -o -iname 'NOTICE*' \) 2>/dev/null)
done < <(echo "$CHECK_OUT" | awk '/^DISTINFO /{print}')

echo "building Automix.app ..."
"$BUILD_PYTHON" -m PyInstaller \
  --noconfirm --clean --windowed \
  --name Automix \
  --paths "$SRC_ROOT" \
  --distpath "$DIST_DIR" \
  --workpath "$WORK_DIR" \
  --specpath "$SPEC_DIR" \
  --add-data "$SRC_ROOT/web:web" \
  --add-data "$REPO_ROOT/presets:presets" \
  --add-data "$REPO_ROOT/schemas/mix_plan.schema.json:schemas" \
  --collect-all jsonschema \
  --collect-all jsonschema_specifications \
  --collect-all referencing \
  --hidden-import jsonschema \
  --hidden-import jsonschema.validators \
  --hidden-import jsonschema_specifications \
  --hidden-import referencing \
  --hidden-import referencing.jsonschema \
  --hidden-import rpds \
  --hidden-import attrs \
  ${ADD_DATA_ARGS[@]+"${ADD_DATA_ARGS[@]}"} \
  ${COPY_METADATA_ARGS[@]+"${COPY_METADATA_ARGS[@]}"} \
  "$SRC_ROOT/app.py" > "$BUILD_LOG" 2>&1 || {
    echo "build failed; see $BUILD_LOG" >&2
    tail -20 "$BUILD_LOG" >&2
    exit 1
  }

APP="$DIST_DIR/Automix.app"
if [[ ! -d "$APP" ]]; then
  echo "the build produced no app at $APP; see $BUILD_LOG" >&2
  exit 1
fi
APP_ABS="$(cd "$(dirname "$APP")" && pwd)/$(basename "$APP")"
echo "built: $APP_ABS ($(du -sh "$APP" | cut -f1))"

# 4. Smoke test: the bundle must be able to check a plan it was given.
#    No server, no browser, no audio -- the App is asked to validate three
#    plans and nothing else.  The fixtures live in a folder this script owns
#    and it cleans up exactly that folder, nothing else.
SMOKE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/automix-smoke-XXXXXX")"
cleanup() {
  case "$SMOKE_DIR" in
    "${TMPDIR:-/tmp}"/automix-smoke-*)
      rm -rf "$SMOKE_DIR"
      ;;
    *)
      echo "leaving $SMOKE_DIR alone: it is not the folder this build made" >&2
      ;;
  esac
}
trap cleanup EXIT

"$BUILD_PYTHON" build_smoke.py --dir "$SMOKE_DIR" --binary "$APP/Contents/MacOS/Automix"
