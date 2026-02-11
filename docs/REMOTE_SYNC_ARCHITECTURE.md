# Remote Sync Architecture: Federated Note Backup & Import

## Overview

A system for backing up notes to private git remotes and importing notes from other machines — conflict-free by design.

Each machine **publishes** its own notes to its own private remote (one writer, no conflicts possible) and **subscribes** to other machines' remotes as read-only imports (pull only, never push).

```
Machine A                          Machine B
┌─────────────────┐               ┌─────────────────┐
│ notes/       (rw) │──push──→ repo-a ←──pull──│ imports/a/   (ro) │
│ imports/b/   (ro) │──pull──→ repo-b ←──push──│ notes/       (rw) │
└─────────────────┘               └─────────────────┘
```

## Motivation

- Notes should survive machine failure (backup to private remote)
- Knowledge should be accessible across machines without merge conflicts
- The system should batch pushes intelligently, not spam the remote on every write

## Part 1: Batched Push with Debounce

### Concept

Every note write already creates a local git commit. Rather than pushing after each commit (wasteful) or requiring manual pushes (easy to forget), use a **debounce timer**:

```
note write → start 120s timer
  another write within window → reset to 60s
  another write within window → reset to 60s
  ...
  60s of silence → push all accumulated commits
```

This naturally batches bursts of activity. A planning session that creates 20 notes results in one push, not twenty.

### Implementation Details

- Background thread with a resettable timer
- The service layer signals the sync thread after each git commit
- If the MCP server shuts down with pending commits, it does a final flush push
- `zk_system action="git_push"` for manual push (bypasses timer)
- Push failures are logged, not raised — notes are safely committed locally regardless

### Why Batching Matters

- Avoids redundant network round-trips and SSH handshakes during active sessions
- GitHub's soft limit is ~100 operations/hour for authenticated git — not a practical concern with batching, but good hygiene regardless
- Each push has overhead; one push of 20 commits is far cheaper than 20 pushes of 1 commit

## Part 2: Federated Note Imports

### The Problem with Shared Repos

**Single shared repo** (everyone pushes to one repo): Merge conflicts are inevitable. Two machines create notes offline, both push — now you need merge strategies, conflict resolution UI, and recovery flows. It's a distributed systems problem masquerading as a file sync problem.

**Branch-per-machine in one repo**: Still needs a read strategy for combining branches. Force-pushes are tempting and dangerous.

### The Solution: Separate Repos, Read-Only Imports

Each machine has its own remote. Other machines clone those remotes as read-only imports. Zero conflicts possible because no machine ever writes to another machine's repo.

This is the same pattern as RSS feeds, Secure Scuttlebutt, or how git itself is designed at the protocol level — independent repos with one-way data flow.

### Directory Structure

```
~/.zettelkasten/
  notes/                    # YOUR notes — read-write, pushed to your remote
    general/
    project-a/
  imports/                  # Other machines' notes — read-only, pulled
    work-desktop/           # git clone of work-desktop's remote
      general/
      project-b/
    server/                 # git clone of server's remote
      ...
  .env                      # config: your remote, import sources, machine identity
  zettelkasten.db
```

### Key Design Decisions

**Identity.** Each installation needs a name (`ZETTELKASTEN_MACHINE_ID=laptop`). This becomes the directory name under `imports/` on other machines and gets embedded in status/search results so you know where a note came from.

**Indexing imports.** Imported notes are in the database and FTS index so they're searchable. They are flagged as imported and read-only — you can search them, link to them, read them, but `zk_update_note` and `zk_delete_note` refuse to touch them. The in-memory DB (current default) rebuilds from markdown on startup, so it scans both `notes/` and `imports/*/` during rebuild, tagging each note with its source.

**Import freshness.** Pull imports on startup (catches overnight changes from other machines) and on-demand via `zk_system action="pull_imports"`. Periodic background pull is possible but probably overkill — startup + manual is enough.

**Note ID collisions.** UUIDs handle this — two machines creating notes independently won't collide. Project names will overlap (both machines might have a `general/` project). The import namespace (machine name) disambiguates: search results show `work-desktop:general/Some Note` vs your local `general/Some Note`.

**Cross-machine links.** You can link a local note to an imported note. The link stores the note ID (UUID), which is globally unique. If the imported note is later deleted upstream, the link becomes dangling — same as deleting a local note that has inbound links, which znote-mcp already handles.

**Storage.** Each import is a full clone. For a zettelkasten with thousands of markdown files, this is negligible (markdown is tiny). Shallow clones (`git clone --depth 1`) can reduce it further if history isn't needed on the import side.

**Auth.** SSH deploy keys are the cleanest path. Each machine generates a key, you add it as a read-only deploy key to the repos you want to import. Your own repo gets a read-write deploy key. No passwords, no credential helpers, no token rotation.

## Configuration

In `~/.zettelkasten/.env`:

```bash
# Your machine identity (used as namespace in imports on other machines)
ZETTELKASTEN_MACHINE_ID=laptop

# Your notes remote (push target)
ZETTELKASTEN_GIT_REMOTE=git@github.com:user/zk-laptop.git

# Push behavior: debounce delay in seconds (0 = push after every commit)
ZETTELKASTEN_GIT_PUSH_DELAY=120
ZETTELKASTEN_GIT_PUSH_EXTEND=60

# Import sources (JSON: name -> remote URL)
ZETTELKASTEN_IMPORTS={"work-desktop": "git@github.com:user/zk-work.git", "server": "git@github.com:user/zk-server.git"}

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
- Has no remote operations whatsoever — strictly local

The remote sync feature builds on top of this by adding push/pull operations and a sync thread.

## Implementation Phases

### Phase 1: Push Your Own Notes (Debounce)

- `ZETTELKASTEN_GIT_REMOTE`, `ZETTELKASTEN_GIT_PUSH_DELAY`, `ZETTELKASTEN_GIT_PUSH_EXTEND` config options
- `GitSyncService` with debounce thread: receives commit signals, manages timer, executes push
- On startup: auto-configure remote (`git remote add/set-url origin <url>`)
- On shutdown: flush push if pending commits
- `zk_system action="git_push"` for manual push
- `zk_status` reports: remote URL, last push time, pending commit count

### Phase 2: Import Other Machines' Notes

- `ZETTELKASTEN_MACHINE_ID`, `ZETTELKASTEN_IMPORTS`, `ZETTELKASTEN_IMPORT_ON_STARTUP` config options
- Import directory management: clone if missing, pull if exists
- Index imported notes in DB as read-only (with source machine tag)
- Reject write operations on imported notes
- `zk_system action="pull_imports"` for manual refresh
- Search integration: imported notes appear in search results with source attribution

### Phase 3: Polish

- Cross-machine link support in `zk_create_link`
- Import freshness reporting in `zk_status` (last pull time per source)
- `zk_list_notes mode="imports"` to browse imported notes by source
- Shallow clone option for imports (`ZETTELKASTEN_IMPORT_SHALLOW=true`)
- SSH key setup guidance in `.env.example`
- Handling of import source removal (cleanup orphaned imports)
