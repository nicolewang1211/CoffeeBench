# Checkpoint System for API Resilience

This guide explains how to use checkpoints to resume experiments after API failures.

## What Are Checkpoints?

Checkpoints automatically save your experiment progress after each day. If the API fails or crashes, you can resume from the last completed day instead of restarting from Day 0.

## How It Works

**Automatic saving:**
- After each day completes, the system saves a checkpoint
- Checkpoint includes: current day, agent states, economic data
- Saved to: `trajectories/<experiment_name>/seed_<N>/checkpoint.json`

**Resuming after failure:**
- If experiment crashes, checkpoint remains
- Restart with `--resume` flag
- System loads checkpoint and continues from last completed day

## Usage

### Running with Checkpoints (Default)

Checkpoints are enabled by default:

```bash
# Run experiment - checkpoints saved automatically
python -m coffeebench.main --config experiments/roaster_focal_sonnet.toml --seed 0
```

If the experiment crashes (API failure, power outage, etc.), resume with:

```bash
# Resume from last checkpoint
python -m coffeebench.main --config experiments/roaster_focal_sonnet.toml --seed 0 --resume
```

### Disabling Checkpoints

If you don't want checkpoints (e.g., for faster runs without I/O overhead):

```bash
# Run without checkpoints
python -m coffeebench.main --config experiments/roaster_focal_sonnet.toml --seed 0 --no-checkpoint
```

### Starting Fresh

To ignore existing checkpoint and start from Day 0:

```bash
# Delete checkpoint and start fresh
rm trajectories/roaster_focal_sonnet/seed_0/checkpoint.json
python -m coffeebench.main --config experiments/roaster_focal_sonnet.toml --seed 0
```

Or use the `--force-restart` flag:

```bash
# Automatically delete checkpoint and start fresh
python -m coffeebench.main --config experiments/roaster_focal_sonnet.toml --seed 0 --force-restart
```

## Example Workflow with Unstable API

**Scenario:** Intenext API goes down mid-experiment

1. **Start experiment:**
   ```bash
   python -m coffeebench.main --config experiments/roaster_focal_sonnet.toml --seed 0
   ```

2. **Experiment runs:** Days 0-25 complete successfully

3. **API fails:** Day 26 crashes due to API timeout after all retries

4. **Resume when API is back:**
   ```bash
   # Wait for API to stabilize, then resume
   python -m coffeebench.main --config experiments/roaster_focal_sonnet.toml --seed 0 --resume
   ```

5. **Continues from Day 26:** No need to re-run Days 0-25

## Multi-Provider Fallback (Alternative Approach)

If you have multiple API providers (Intenext + Gemini), you can configure fallback:

**Option 1: Use Gemini as backup**

Edit `.env`:
```bash
# Primary: Intenext
OPENAI_API_KEY="intenext-key"
OPENAI_BASE_URL="https://api.intenext.ai/v1"

# Backup: Direct Gemini
GEMINI_API_KEY="your-gemini-key"
```

Create a config that uses Gemini:
```toml
# experiments/roaster_focal_gemini.toml
[models]
default = "gemini-2.0-flash-exp"
roaster_A = "gemini-2.0-flash-exp"
```

**If Intenext fails completely:**
1. Let experiment crash
2. Switch to Gemini config
3. Resume from checkpoint:
   ```bash
   python -m coffeebench.main --config experiments/roaster_focal_gemini.toml --seed 0 --resume
   ```

**Option 2: Manual provider switching**

If API fails mid-run:
1. Stop experiment (Ctrl+C)
2. Update `.env` to use different provider
3. Resume with `--resume`

## Checkpoint File Format

Checkpoints are saved as JSON:

```json
{
  "day": 25,
  "state": {
    "agents": {...},
    "ledger": {...},
    "market": {...}
  }
}
```

**Location:** `trajectories/<experiment_name>/seed_<N>/checkpoint.json`

## Troubleshooting

### Checkpoint won't load

**Error:** "Failed to load checkpoint"

**Solutions:**
- Checkpoint file may be corrupted
- Delete it and start fresh: `rm trajectories/.../checkpoint.json`
- Use `--force-restart` flag

### Experiment keeps failing at same day

**Issue:** API consistently fails at specific day (e.g., Day 50)

**Possible causes:**
- Context window too large (too much history)
- Specific agent state causes API error

**Solutions:**
1. Check logs for error details
2. Try different model with larger context window
3. Reduce `max_days` in config to avoid problematic day

### Checkpoint overhead

**Issue:** Checkpoints slow down experiment

**Solution:**
- Disable checkpoints: `--no-checkpoint`
- Only use for long runs (90 days) where API stability is uncertain
- For short runs (30 days), overhead may not be worth it

## Best Practices

1. **Always use checkpoints for 90-day runs** - Too much time/cost to lose
2. **Keep checkpoints for unstable APIs** - Intenext, free tiers, rate-limited providers
3. **Disable for stable APIs** - Direct OpenAI/Anthropic with paid accounts
4. **Monitor checkpoint file size** - Should be < 10MB; if larger, may indicate issues
5. **Clean up after success** - Checkpoints auto-delete on completion

## Summary

**For your Intenext setup:**

```bash
# Run with automatic checkpointing (recommended)
python -m coffeebench.main --config experiments/roaster_focal_sonnet.toml --seed 0

# If it crashes, resume from last checkpoint
python -m coffeebench.main --config experiments/roaster_focal_sonnet.toml --seed 0 --resume

# If Intenext is down completely, switch to Gemini
python -m coffeebench.main --config experiments/roaster_focal_gemini.toml --seed 0 --resume
```

This gives you maximum resilience against API instability while minimizing lost progress.
