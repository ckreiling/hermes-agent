# Omarchy branch

`omarchy` is the maintained local-customization branch for
<https://github.com/ckreiling/hermes-agent/tree/omarchy>. Keep fixes committed on
this branch rather than as edits to generated builds or installed plugins.

## What this branch changes

The Ink TUI (`hermes --tui`) reports its lifecycle over OSC 7501 so native Herdr
can list it as an agent. Reports originate in the terminal-owning frontend,
not the shared gateway. The setting `display.program_status` can disable them.
The classic CLI is not changed by this patch.

Related upstream work: NousResearch/hermes-agent#138147, #140837, and #149976.
When upstream ships equivalent TUI reporting, review and remove our overlapping
implementation instead of maintaining two reporters.

## Git remotes on the installed checkout

This installation retains its existing remote names deliberately:

- `origin`: `git@github.com:NousResearch/hermes-agent.git` — official update source.
- `fork`: `git@github.com:ckreiling/hermes-agent.git` — our published branch.
- `branch.omarchy.pushRemote` and `remote.pushDefault`: `fork`.

Do not change `origin` to the fork without revisiting the updater policy below:
Hermes's in-place updater merges **origin/main**, not the branch's Git tracking
remote. The branch's tracking target can still be `fork/omarchy` for publishing.

## Upgrading this machine

The installed configuration uses Hermes's existing upgrade machinery:

- This install's update channel is `main` (not a detached stable-release pin).
- `updates.parked_branch_strategy` is `update_in_place`.

From a separate terminal, with no important agent work running:

```bash
cd ~/.hermes/hermes-agent
git status --short
# Commit or explicitly stash any uncommitted changes first.
hermes update --check
hermes update
# Confirm the patch still works in a newly opened Herdr session.
git branch --show-current
git push fork omarchy
```

On a clean `omarchy` branch with custom commits, `hermes update` merges official
`origin/main` into this branch rather than switching to `main`. It prepares
managed dependencies, refreshes frontend products, and restarts the gateways
through the normal updater. It does not publish the resulting merge to our
fork; push only after validation. A fresh TUI process is needed to load a new
compiled frontend; a gateway restart alone does not replace an already-running
Node TUI.

**Do not use `hermes update --branch omarchy` to fetch official upgrades.** That
asks the updater for `origin/omarchy`, which does not exist in the official
repository. Likewise, `--switch-branch` or choosing the `stable` channel can leave
this customized branch; those are intentional transitions, not this branch's
normal upgrade path.

The in-place updater refuses a dirty parked branch and stops on merge conflicts
rather than silently dropping the local patch. Preserve local changes before an
upgrade. If the updater reports a conflict, inspect its receipt and repository
state, resolve the upstream change on `omarchy`, rerun the applicable tests and
build, and use that revision's supported PM/update procedure. Do not force-reset
`omarchy` to `origin/main` or use `git clean` to make an update proceed.

## Setting up the policy on another source installation

After checking out this branch, configure the same remote layout above, then:

```bash
git config branch.omarchy.pushRemote fork
git config remote.pushDefault fork
hermes update --set-channel main
hermes config set updates.parked_branch_strategy update_in_place
```

In this revision, `--set-channel` only saves this install's channel; it does not
apply an upgrade. Verify with `hermes config get update.installs` and
`hermes config get updates.parked_branch_strategy`.

## Validation and recovery

Run the TUI's tests and build after changing the patch:

```bash
pnpm --dir ui-tui test
pnpm --dir ui-tui typecheck
pnpm --dir ui-tui build
```

In a native Herdr pane, `herdr-core get <terminal> --json` should show a Hermes
status after TUI startup, working during a turn, and completion afterward. Quit
back to a shell and confirm that the program-status record is cleared.

Hermes's updater records pre-update recovery refs/tags and receipts, but code
rollback is not database rollback. Back up user state before substantial
upgrades; do not run old code on a state database migrated by a newer release.
