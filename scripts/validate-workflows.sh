#!/usr/bin/env bash

# This script is used as a pre-commit hook to ensure that no unauthorized 
# workflow files are accidentally committed.

ALLOWED_WORKFLOWS=(
    "build-rockchip.yml"
    "auto-sync-controller.yml"
    "tqp-release.yml"
    "rebuild-on-push.yml"
)

if [ ! -d ".github/workflows" ]; then
    exit 0
fi

# Get all tracked or staged files in .github/workflows
FILES=$(git ls-files .github/workflows/ | grep "\.yml$")

FAILED=0

for file in $FILES; do
    filename=$(basename "$file")
    
    # Check if the filename is in the allowed list
    is_allowed=0
    for allowed in "${ALLOWED_WORKFLOWS[@]}"; do
        if [[ "$filename" == "$allowed" ]]; then
            is_allowed=1
            break
        fi
    done
    
    if [[ $is_allowed -eq 0 ]]; then
        echo "ERROR: Unauthorized workflow file found: $file"
        FAILED=1
    fi
done

if [[ $FAILED -eq 1 ]]; then
    echo ""
    echo "Per AGENTS.md, only the following custom workflows are allowed:"
    for allowed in "${ALLOWED_WORKFLOWS[@]}"; do
        echo "  - $allowed"
    done
    echo ""
    echo "Please run: git rm <unauthorized_files> to remove them."
    exit 1
fi

exit 0
