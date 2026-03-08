#!/usr/bin/env bash
# Update cargoHash in flake.nix when Cargo.lock changes
set -euo pipefail

echo "Updating cargoHash in flake.nix..."

# Build and capture the new hash from error output
echo "Building to get new hash..."
new_hash=$(nix build 2>&1 | grep "got:" | head -1 | sed 's/.*got: *//g' | tr -d ' ')

if [ -z "$new_hash" ]; then
    echo "No hash update needed - build succeeded"
    exit 0
fi

echo "Found new hash: $new_hash"

# Get current hash from flake.nix
current_hash=$(grep -o 'cargoHash = "sha256-[^"]*"' flake.nix | head -1 | sed 's/cargoHash = "//g' | sed 's/"//g')

if [ -z "$current_hash" ]; then
    echo "Could not find current cargoHash in flake.nix"
    exit 1
fi

echo "Replacing $current_hash with $new_hash"

# Replace all occurrences of the old hash with the new one
escaped_current=$(echo "$current_hash" | sed 's/[[\.*^$()+?{|]/\\&/g')
escaped_new=$(echo "$new_hash" | sed 's/[[\.*^$()+?{|]/\\&/g')

sed -i "s/$escaped_current/$escaped_new/g" flake.nix

echo "Hash updated in flake.nix"

# Verify the build now works
echo "Verifying build..."
if nix build 2>/dev/null; then
    echo "Build successful with new hash!"
else
    echo "Build failed even with new hash"
    exit 1
fi

echo "cargoHash successfully updated!"
echo ""
echo "Commit the changes:"
echo "  git add flake.nix"
echo "  git commit -m \"chore: update cargoHash\""
