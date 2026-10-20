#!/usr/bin/env bash
set -euo pipefail
zip -r submission.zip . -x '.*' '*/.*' 'runs/**' 'data/**' 'submission.zip'
