# How to Continue This in Claude Code

*The answer to: "Let us move to Claude Code first — HOW do we do that?"*

Claude Code is a command-line agentic coding tool. It reads `CLAUDE.md` from the
repo root automatically at the start of every session, giving it full project context
without you having to re-explain anything.

---

## 1. Install Claude Code (one time)

```bash
# Requires Node.js 18+ and an Anthropic API key
npm install -g @anthropic-ai/claude-code

# Set your API key
export ANTHROPIC_API_KEY="sk-ant-..."    # or add to ~/.bashrc / ~/.zshrc

# Verify
claude --version
```

Or in VS Code / Cursor, install the **Claude Code** extension.

---

## 2. Start a Session in This Repo

```bash
# Clone the repo (if not already done)
git clone https://github.com/diazMelgarejo/agentra-dingbot.git
cd agentra-dingbot

# Start Claude Code
claude

# Claude Code will automatically read CLAUDE.md and know the full project context.
# You can immediately say: "continue the work on the feature branch"
```

---

## 3. What Claude Code Reads Automatically

When you `claude` from inside this repo, it reads:
- **`CLAUDE.md`** (root) — project state, commands, architecture, TDD rules
- **`.claude/settings.json`** (if present) — permissions, allowed tools
- **`.gitignore`** — knows what to exclude

No need to paste context. It already knows:
- All 8 build steps and their status
- The critical architecture rules (never `dataclasses.asdict()`, etc.)
- How to run tests, backtest, snapshot export
- Open PRs and what's pending

---

## 4. Useful Starting Prompts for Claude Code

```bash
# Start Claude Code
claude

# Then say any of these:
> "What's the current test status?"
> "Wire the real Polymarket markets into the WebSocket payload"
> "Run the Playwright smoke tests"
> "Open the PR and merge the feature branch"
> "Show me what's left in docs/FUTURE_PLANS_NEXT.md"
> "Run the council in demo mode"
> "Make a new branch and fix the risk_reward_ratio in the dashboard"
```

---

## 5. Give Claude Code Tool Permissions

For this repo, Claude Code needs:
- `bash` — to run tests, generate snapshots, push to git
- `file-read`, `file-write` — to read/write source files
- `web-search` — to look up library docs

Claude Code will ask permission the first time. Or pre-allow with:
```bash
# .claude/settings.json (create this at the repo root)
{
  "allowedTools": ["bash", "file-read", "file-write", "web-search"],
  "env": {
    "PYTHONPATH": "src",
    "LLM_PROVIDER": "none",
    "FREQTRADE_MODE": "off"
  }
}
```

---

## 6. Resume the Current Feature Branch

```bash
cd agentra-dingbot
git checkout 2026-06-20-snapshot-council-livebind
claude

# Claude Code will see the branch and know what was built.
# Say: "Show me what's different from main and let's finish it"
```

---

## 7. What You Get Over the Web UI

| Capability | Web UI (claude.ai) | Claude Code |
|---|---|---|
| File read/write | ✅ (via uploads) | ✅ (native, any file) |
| Run bash commands | ✅ (via tools) | ✅ (native) |
| Multi-file edits | Manual uploads | ✅ Automatic |
| Persistent git history | Must be rebuilt | ✅ Uses local git |
| Run tests automatically | ✅ | ✅ |
| MCP server connections | ✅ | ✅ |
| Context auto-load | ✅ (from claude.ai) | ✅ (reads CLAUDE.md) |
| Session continuity | Must re-paste context | ✅ CLAUDE.md is always loaded |

---

## 8. Using the orama-system with Claude Code

The `CLAUDE.md` file already triggers the oramasys-method when you prefix requests
with "oramasys" or "ultrathink":

```bash
claude
> "oramasys: implement the portfolio_agent LangGraph node"
> "ultrathink: review the risk manager for edge cases"
```

---

## 9. GitHub PAT — Never Paste It Again

Add it as an env var in your shell or `.claude/settings.json`:
```bash
export GITHUB_PAT="github_pat_..."    # in ~/.zshrc or ~/.bashrc
```

Then in Claude Code:
```bash
git remote set-url origin "https://diazMelgarejo:${GITHUB_PAT}@github.com/..."
```

Or use GitHub CLI (`gh`) instead of PAT:
```bash
gh auth login    # one-time setup
gh pr create --title "..." --body "..."   # much easier than PAT
```

---

## 10. Install GitHub CLI (Optional but Recommended)

```bash
# macOS
brew install gh

# Linux
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update && sudo apt install gh

# Authenticate
gh auth login   # follows browser prompt

# Then open PRs, merge, list issues from terminal:
gh pr create --fill
gh pr merge 2 --squash
```
