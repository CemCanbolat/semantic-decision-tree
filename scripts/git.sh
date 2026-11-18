#!/usr/bin/env bash
# git wrapper: the workspace's .git path is a read-only mount in the
# build sandbox, so the repository lives in .git-real
exec git --git-dir="$PWD/.git-real" --work-tree="$PWD" "$@"
