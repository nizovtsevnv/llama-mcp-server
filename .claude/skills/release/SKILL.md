---
name: release
description: Create a release — quality checks, version bump, changelog, tag, push
user-invocable: true
allowed-tools: Read, Edit, Write, Grep, Glob, Bash(git *), Bash(nix develop *), Bash(cargo *), Bash(gh *), AskUserQuestion
---

# Release Management

Create a release for whisper-mcp-server: quality checks, version bump, changelog update, commit, tag, push, GitHub release.

## Prerequisites

- Clean or intentionally dirty working directory
- All tests passing locally
- `gh` CLI authenticated

## Workflow

### Step 1: Pre-release quality checks

Run in sequence. If ANY fail — stop and report errors:

```bash
nix develop -c cargo fmt --check
nix develop -c cargo clippy -- -D warnings
nix develop -c cargo test
```

If `cargo fmt --check` fails — suggest running `nix develop -c cargo fmt`.
If clippy or tests fail — show errors, abort release.

### Step 2: Change analysis

Gather changes from two sources:

**Uncommitted changes:**
```bash
git status --porcelain
git diff --stat
```

**Commits since last tag:**
```bash
last_tag=$(git describe --tags --abbrev=0 2>/dev/null || echo "")
if [ -n "$last_tag" ]; then
    git log ${last_tag}..HEAD --oneline --no-decorate
else
    git log --oneline --no-decorate
fi
```

Categorize commits by conventional commit type:
- `feat:` → Added
- `fix:` → Fixed
- `refactor:`, `perf:` → Changed
- `docs:` → Changed (if user-facing)
- `chore:` → usually omit unless significant

Save the analysis for changelog generation in step 6.

### Step 3: Version detection

Read current version from Cargo.toml (line 3, the single source of truth — flake.nix reads it via `builtins.fromTOML`).

Also get last git tag:
```bash
git describe --tags --abbrev=0 2>/dev/null || echo "none"
```

If Cargo.toml version and last tag differ (accounting for `v` prefix), report the mismatch and ask the user which is correct.

### Step 4: Version selection

Use AskUserQuestion to present options:

```
Current version: X.Y.Z

Select release type:
1. patch (X.Y.Z → X.Y.(Z+1)) — bug fixes, minor changes
2. minor (X.Y.Z → X.(Y+1).0) — new features, backwards compatible
3. major (X.Y.Z → (X+1).0.0) — breaking changes
4. custom — enter specific version
5. recreate X.Y.Z — recreate existing tag (for failed CI/CD)
```

For **recreate**: confirm with user that this will delete and recreate the tag.

### Step 5: Update version

**Cargo.toml** — update `version = "X.Y.Z"` on line 3 using Edit tool.

Then update Cargo.lock:
```bash
nix develop -c cargo check
```

**Always** update cargoHash after version bump — the nix vendor hash changes even if dependencies didn't change, because the package version is part of the derivation.

Run `nix build` to get the new hash from the error output:
```bash
new_hash=$(nix build 2>&1 | grep "got:" | head -1 | sed 's/.*got: *//g' | tr -d ' ')
```

If `nix build` succeeds (no hash mismatch) — cargoHash is already correct, skip.
If it fails with a hash mismatch — update `cargoHash` in `flake.nix` with the new hash using Edit, then verify with `nix build`.

After updates, verify no old version is hardcoded anywhere it shouldn't be:
```bash
grep -rn "OLD_VERSION" --include="*.rs" --include="*.toml" --include="*.md" --include="*.yml" --exclude-dir=target .
```

Expected matches (ignore these):
- `Cargo.toml` — source of truth
- `Cargo.lock` — auto-updated
- `CHANGELOG.md` — release history (old versions are fine)

Unexpected matches to **fix**:
- `.rs` files — use `env!("CARGO_PKG_VERSION")` instead of hardcoded strings
- `.yml` CI files — should not contain version numbers
- `README.md` — use `<version>` placeholder in protocol examples, not a real version

### Step 6: CHANGELOG.md

If CHANGELOG.md doesn't exist, create it with header:

```markdown
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
```

Generate a new section from step 2 analysis. **IMPORTANT**: the changelog MUST cover ALL commits since the last tag (or since the first commit if no tags exist) AND any uncommitted changes. Do not limit the changelog to only uncommitted changes — include every meaningful commit from the range collected in step 2.

```markdown
## [vX.Y.Z] - YYYY-MM-DD

### Added
- ...

### Changed
- ...

### Fixed
- ...
```

Only include non-empty subsections. Use the `v` prefix in headers to match tag format.

Show the draft to the user via AskUserQuestion:
1. Use as-is
2. Let me edit it (pause for manual edits)
3. Regenerate

Insert the new section after the header (before the first existing `## [v` entry).

Add release link at the bottom of the file:
```markdown
[vX.Y.Z]: https://github.com/nizovtsevnv/whisper-mcp-server/releases/tag/vX.Y.Z
```

### Step 7: Post-update quality checks

Run the same checks as step 1:

```bash
nix develop -c cargo fmt --check
nix develop -c cargo clippy -- -D warnings
nix develop -c cargo test
```

If any fail after the version/changelog updates — report and abort.

### Step 8: Commit + tag + push

**Stage and commit:**
```bash
git add -A
git commit -m "chore: release vX.Y.Z

<summary of key changes from changelog>"
```

Never use `--no-verify`. If pre-commit hooks fail — fix issues and create a new commit.

**Create annotated tag** (with `v` prefix to match release.yml trigger):
```bash
git tag -a vX.Y.Z -m "Release vX.Y.Z"
```

For **recreate** mode, first delete existing tag:
```bash
git tag -d vX.Y.Z
git push origin :refs/tags/vX.Y.Z 2>/dev/null || true
```

**Show summary and ask before push** via AskUserQuestion:
```
Ready to push release vX.Y.Z

This will:
- Push commit and tag to origin
- Create or update GitHub Release via gh CLI
- Trigger CI/CD which builds binaries for 5 platforms

Push now? [yes / no / show undo instructions]
```

**If yes:**
```bash
git push && git push origin vX.Y.Z
```

Extract changelog body and create or update GitHub release:
```bash
changelog=$(awk '/^## \[vX\.Y\.Z\]/{flag=1; next} /^## \[/{flag=0} flag' CHANGELOG.md)
if gh release view vX.Y.Z >/dev/null 2>&1; then
  gh release edit vX.Y.Z --title "vX.Y.Z" --notes "$changelog" --draft=false
else
  gh release create vX.Y.Z --title "vX.Y.Z" --notes "$changelog"
fi
```

If a release already exists (e.g. created by CI), update it. Otherwise create a new one. CI will attach binaries.

**If no**, show manual push commands and undo instructions:
```
To push later:
  git push && git push origin vX.Y.Z

To undo:
  git tag -d vX.Y.Z
  git reset --hard HEAD^
```

**Final report:**
```
Release vX.Y.Z created.

- Commit: <hash> chore: release vX.Y.Z
- Tag: vX.Y.Z
- Release: https://github.com/nizovtsevnv/whisper-mcp-server/releases/tag/vX.Y.Z
- CI/CD: https://github.com/nizovtsevnv/whisper-mcp-server/actions

CI will build binaries for: linux-x86_64, linux-x86_64-musl, windows-x86_64, macos-x86_64, macos-arm64
```

## Error Handling

### Uncommitted changes at start

If `git status --porcelain` shows changes before step 1, ask user:
1. Include in release commit
2. Commit separately first (pause release)
3. Stash and continue
4. Cancel release

### Tag already exists (non-recreate mode)

Offer to switch to recreate mode or pick a different version.

### Network/push failures

Show error, provide manual push and undo commands.

## Rules

- Tag format: `vX.Y.Z` (with `v` prefix) — matches release.yml trigger `push: tags: ['v*']`
- Version in Cargo.toml: `X.Y.Z` (without `v` prefix)
- flake.nix reads version from Cargo.toml automatically — do not edit it for version bumps
- Commit message in English, conventional commits format
- Never use `--no-verify`
- Communicate with the user in their language
