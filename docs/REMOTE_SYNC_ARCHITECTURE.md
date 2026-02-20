# Remote Sync Architecture: Shared Repo with Sparse Checkout

## Overview

A system for syncing notes across team members through a shared GitHub repository — conflict-free by design.

Each user owns a directory in the shared repo (`notes/<username>/`), works on a personal branch, and merges to `main` via pull request. Other users' notes are imported from `main` as read-only. Local `~/.zettelkasten/` remains the source of truth for each user.

```
User A                              GitHub (komi-zone)                    User B
┌──────────────┐                    ┌──────────────────┐                 ┌──────────────┐
│ notes/   (rw) │──push──→ branch: │ main             │ ←push──        │ notes/   (rw) │
│ imports/     │          alex/    │   notes/          │    bob/        │ imports/     │
│   bob/   (ro) │←─pull─── notes   │     alex/        │    notes ──pull→│   alex/  (ro) │
│   carol/ (ro) │         from     │     bob/          │   from         │   carol/ (ro) │
└──────────────┘         main      │     carol/        │   main         └──────────────┘
                                   └────────┬─────────┘
                                            ↑
                                     PRs + CODEOWNERS review
```

## Motivation

- Notes should be backed up to a remote and accessible across team members
- Each user should control their own notes without risk of conflicts
- The PR workflow provides review gates and a clean audit trail
- The shared repo avoids proliferating per-user repositories

## How It Works

### Shared Repo Structure

The komi-zone repo contains a `notes/` directory with a subdirectory per user:

```
komi-zone/
  marketplace/              # shared marketplace code, plugins, etc.
  notes/
    alex/                   # alex's notes
      general/
      project-a/
    bob/                    # bob's notes
      general/
    carol/                  # carol's notes
      ...
  .github/
    CODEOWNERS              # enforces per-user write ownership
    workflows/
      notes-guard.yml       # CI check for ownership violations
```

### Per-User Branches

Each user works on their own branch:

| User  | Branch         | Pushes to         | Imports from |
|-------|----------------|-------------------|--------------|
| alex  | `alex/notes`   | `notes/alex/`     | `main`       |
| bob   | `bob/notes`    | `notes/bob/`      | `main`       |
| carol | `carol/notes`  | `notes/carol/`    | `main`       |

**Push flow:** The sync service copies your local notes into the sparse checkout, commits to your branch, and pushes. You own this branch — no coordination needed.

**Merge flow:** When ready, you create a PR from your branch to `main`. CODEOWNERS ensures only you can approve changes to your directory. Colleagues can review if desired.

**Import flow:** Pull `main` to get the latest merged state of everyone's notes. `main` is the blessed snapshot — only reviewed, merged notes appear in imports.

### Sparse Checkout + Partial Clone

Users don't need the full repo on disk. Git's sparse checkout with partial clone downloads only the directories you care about:

```bash
# One-time setup — clones repo metadata without downloading all blobs
git clone --filter=blob:none --sparse git@github.com:org/komi-zone.git ~/.zettelkasten/.remote
cd ~/.zettelkasten/.remote

# Only materialize the notes directories (not marketplace code, etc.)
git sparse-checkout set notes/alex notes/bob notes/carol
```

- `--filter=blob:none` — blob content is fetched on demand, only for paths in the sparse set
- `git sparse-checkout set` — controls which directories appear in the working tree
- Adding a new team member: `git sparse-checkout add notes/newperson`
- Commit metadata for the whole repo is still fetched, but blob data for excluded paths is not

### Local Directory Structure

```
~/.zettelkasten/
  notes/                    # YOUR notes — source of truth, read-write
    general/
    project-a/
  imports/                  # Other users' notes — read-only, pulled from main
    bob/
      general/
    carol/
      ...
  .remote/                  # Sparse checkout of komi-zone repo (managed by sync service)
    notes/
      alex/                 # Staging area — your notes are copied here before push
      bob/                  # Pulled from main — copied to imports/bob/
      carol/                # Pulled from main — copied to imports/carol/
  .env                      # Config: user identity, repo URL, branch, sync behavior
  zettelkasten.db
```

## Ownership Enforcement

### CODEOWNERS (PR review gate)

```
# .github/CODEOWNERS
/notes/alex/       @alex
/notes/bob/        @bob
/notes/carol/      @carol
```

With branch protection requiring CODEOWNERS approval on PRs to `main`, no one can merge changes to your notes directory without your sign-off.

### CI Guard (belt-and-suspenders)

A GitHub Action that verifies each commit in a PR only modifies the author's own directory:

```yaml
# .github/workflows/notes-guard.yml
name: Notes ownership guard
on:
  pull_request:
    paths:
      - 'notes/**'

jobs:
  check-ownership:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - name: Verify PR only modifies author's directory
        run: |
          PR_AUTHOR="${{ github.event.pull_request.user.login }}"
          CHANGED_OWNERS=$(git diff --name-only origin/main...HEAD \
            | grep '^notes/' \
            | cut -d/ -f2 \
            | sort -u)
          for owner in $CHANGED_OWNERS; do
            if [ "$owner" != "$PR_AUTHOR" ]; then
              echo "ERROR: $PR_AUTHOR modified notes/$owner/ — only the owner can modify their directory"
              exit 1
            fi
          done
          echo "OK: all changes are within notes/$PR_AUTHOR/"
```

This catches mistakes even before CODEOWNERS review kicks in.

## Batched Push with Debounce

Every note write creates a local git commit. Rather than pushing after each commit (wasteful) or requiring manual pushes (easy to forget), use a debounce timer:

```
note write → start 120s timer
  another write within window → reset to 60s
  another write within window → reset to 60s
  ...
  60s of silence → push all accumulated commits to your branch
```

A planning session that creates 20 notes results in one push, not twenty.

### Implementation Details

- Background thread with a resettable timer
- The service layer signals the sync thread after each git commit
- If the MCP server shuts down with pending commits, it does a final flush push
- `zk_system action="git_push"` for manual push (bypasses timer)
- Push failures are logged, not raised — notes are safely committed locally regardless
- Pushes target your personal branch (e.g., `alex/notes`), never `main`

### Why Batching Matters

- Avoids redundant network round-trips during active sessions
- Each push has overhead; one push of 20 commits is far cheaper than 20 pushes of 1 commit
- Keeps the branch history readable rather than cluttered with per-note commits

## Import Behavior

### Indexing

Imported notes are added to the database and FTS index so they're searchable. They are flagged as imported and read-only:

- `zk_search_notes`, `zk_fts_search` — imported notes appear in results with source attribution (e.g., `bob:general/Some Note`)
- `zk_get_note` — works on imported notes, shows source user
- `zk_update_note`, `zk_delete_note` — rejected for imported notes
- `zk_create_link` — you can link your notes to imported notes

### Freshness

- **On startup:** pull `main` in the sparse checkout, copy updated notes to `imports/` (catches changes since last session)
- **On demand:** `zk_system action="pull_imports"` triggers a pull and re-index
- **No background polling** — startup + manual is sufficient; `main` only changes when PRs are merged

### Note ID Collisions

UUIDs handle this — two users creating notes independently won't collide. Project names may overlap (both users might have `general/`). The import namespace (username) disambiguates.

### Cross-User Links

You can link a local note to an imported note. The link stores the note ID (UUID), which is globally unique. If the imported note is later removed upstream, the link becomes dangling — same behavior as deleting a local note that has inbound links.

## Configuration

In `~/.zettelkasten/.env`:

```bash
# Your identity (matches your directory under notes/ in the shared repo)
ZETTELKASTEN_USER_ID=alex

# Shared repo URL
ZETTELKASTEN_SYNC_REPO=git@github.com:org/komi-zone.git

# Your branch name (default: <user_id>/notes)
ZETTELKASTEN_SYNC_BRANCH=alex/notes

# Push behavior: debounce delay in seconds (0 = push after every commit)
ZETTELKASTEN_GIT_PUSH_DELAY=120
ZETTELKASTEN_GIT_PUSH_EXTEND=60

# Users to import from main (comma-separated; empty = import all)
ZETTELKASTEN_IMPORT_USERS=bob,carol

# Pull imports on startup (default: true)
ZETTELKASTEN_IMPORT_ON_STARTUP=true
```

## Current Git Integration (Baseline)

The existing `GitWrapper` in `src/znote_mcp/storage/git_wrapper.py`:

- Shells out to `git` CLI via `subprocess.run()` (no gitpython/pygit2)
- Auto-initializes a git repo in the notes directory on startup
- Commits on every note create/update/delete
- Provides version hashes for optimistic concurrency control
- Handles `index.lock` contention with retry logic
- Has no remote operations — strictly local

The sync feature builds on top of this by adding:
- Sparse checkout management for the shared repo
- Push to personal branch (with debounce)
- Pull from `main` for imports
- Copy operations between `~/.zettelkasten/notes/` and the sparse checkout staging area

## Implementation Phases

### Phase 1: Sparse Checkout Setup & Push

- `ZETTELKASTEN_USER_ID`, `ZETTELKASTEN_SYNC_REPO`, `ZETTELKASTEN_SYNC_BRANCH` config options
- `GitSyncService` manages the sparse checkout in `~/.zettelkasten/.remote/`
- On first run: partial clone + sparse checkout setup, create user branch if needed
- Copy `notes/` → `.remote/notes/<user_id>/`, commit to user branch, push
- Debounce thread for batched pushes
- On shutdown: flush push if pending commits
- `zk_system action="git_push"` for manual push
- `zk_status` reports: repo URL, branch, last push time, pending commit count

### Phase 2: Import from Main

- `ZETTELKASTEN_IMPORT_USERS`, `ZETTELKASTEN_IMPORT_ON_STARTUP` config options
- On startup: fetch + checkout `main` in sparse checkout, copy other users' notes to `imports/<user>/`
- Update sparse-checkout patterns based on `ZETTELKASTEN_IMPORT_USERS`
- Index imported notes in DB as read-only with source user tag
- Reject write operations on imported notes
- `zk_system action="pull_imports"` for manual refresh
- Search integration: imported notes appear with source attribution

### Phase 3: Ownership & Polish

- `.github/CODEOWNERS` template generation
- `.github/workflows/notes-guard.yml` CI check template
- Cross-user link support in `zk_create_link`
- Import freshness reporting in `zk_status` (last pull time, main HEAD)
- `zk_list_notes mode="imports"` to browse imported notes by source user
- Handling of user removal (cleanup orphaned imports)
- `zk_system action="setup_sync"` guided setup wizard (creates branch, configures sparse checkout, validates SSH access)
