#!/usr/bin/env bash
set -euo pipefail

DEMO=${GRADIO_DEMO:-object_placement}

case "${DEMO}" in
  scene_comp)
    TARGET_SCRIPT="gradio_demo_sc.py"
    ;;
  object_placement)
    TARGET_SCRIPT="gradio_demo_op.py"
    ;;
  *)
    echo "Unknown GRADIO_DEMO value: ${DEMO}" >&2
    echo "Use 'scene_comp' or 'object_placement'." >&2
    exit 1
    ;;
esac

exec python "${TARGET_SCRIPT}" "$@"
