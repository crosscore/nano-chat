#!/bin/bash
# sync.sh - nanochatリポジトリをDGX Spark (ssh spark) に同期する
# Usage: bash sync.sh

set -euo pipefail

REMOTE_HOST="spark"
REMOTE_DIR="~/repos/"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)/nanochat/"

echo "🔄 Syncing nanochat to ${REMOTE_HOST}:${REMOTE_DIR} ..."

# リモートにディレクトリを作成
ssh "${REMOTE_HOST}" "mkdir -p ${REMOTE_DIR}"

# rsyncで同期（.git, .venv, __pycache__等は除外）
rsync -avz --progress \
    --exclude='.git/' \
    --exclude='.venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.cache/' \
    --exclude='.claude/' \
    "${LOCAL_DIR}" "${REMOTE_HOST}:${REMOTE_DIR}nanochat/"

echo "✅ Sync complete: ${REMOTE_HOST}:${REMOTE_DIR}nanochat/"
