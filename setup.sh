#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./setup.sh
  ./setup.sh help

This helper prints the lightweight evo-harness onboarding path.
EOF
}

case "${1:-framework}" in
  framework|"")
    echo "Core evo-harness setup is lightweight."
    echo
    echo "The framework itself uses only the Python standard library."
    echo "Install the CLI from this checkout and start with the bundled CPU demo:"
    echo
    echo "  python3 -m pip install -e ."
    echo "  evo-harness demo"
    echo "  evo-harness run \\"
    echo "    --population-size 2 \\"
    echo "    --workers 1 \\"
    echo "    --gpus 0 \\"
    echo "    --dry-run"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage
    exit 1
    ;;
esac
