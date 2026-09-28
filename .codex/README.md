# Codex repository support

Repository skills are stored in [`../.agents/skills`](../.agents/skills), the
directory Codex discovers for project-scoped skills. Each skill has a required
`SKILL.md` and may include `scripts/`, `references/`, `assets/`, or
`agents/openai.yaml`.

`.codex/` is kept for Codex configuration and repository support files. It is
not the canonical skill discovery directory.

To migrate portable Claude skills after an update, run:

```powershell
node scripts/migrate-claude-skills-to-codex.mjs
```

Keep skill names unique across the repository. Restart Codex if a newly added
skill is not detected in the current session.

## Repository hooks

Lifecycle hooks are configured in `hooks.json`; their Python handlers live in
`hooks/`. They provide repository context, sensitive-file blocking, compact
state, tool-call counting, and bounded post-edit checks.

After adding or changing a hook, restart Codex and open `/hooks` to review and
trust its current definition. Test execution remains opt-in through
`"testRunner": true` in `.ck.json`; TypeScript typechecking runs after Codex
edits TypeScript or JavaScript files through `apply_patch`.
