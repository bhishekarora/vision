#!/usr/bin/env bash
# Sync local changes to the configured GitHub repository.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${SYNC_REMOTE:-https://github.com/bhishekarora/vision.git}"
BRANCH="${SYNC_BRANCH:-main}"
MESSAGE="${SYNC_MESSAGE:-Code updated}"

cd "$REPO_ROOT"

git init >/dev/null 2>&1 || true

if git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
    git checkout "$BRANCH" >/dev/null 2>&1 || git switch "$BRANCH" >/dev/null 2>&1
else
    current_branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo)"
    if [[ -z "$current_branch" || "$current_branch" == "HEAD" ]]; then
        git checkout -b "$BRANCH"
    else
        git branch -M "$BRANCH"
    fi
fi

if ! git remote get-url origin >/dev/null 2>&1; then
    git remote add origin "$REMOTE"
fi

default_remote="$(git remote get-url origin 2>/dev/null || echo)"
if [[ "$default_remote" != "$REMOTE" ]]; then
    echo "Updating origin remote to $REMOTE"
    git remote set-url origin "$REMOTE"
fi

status_output="$(git status --porcelain)"
if [[ -z "$status_output" ]]; then
    echo "Nothing to commit. Working tree clean."
    exit 0
fi

git add -A

git commit -m "$MESSAGE"

git push origin "$BRANCH"

echo "Sync complete: pushed to $REMOTE ($BRANCH)."
