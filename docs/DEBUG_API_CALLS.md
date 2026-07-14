# Debugging API Calls and Tool Calling Issues

This guide explains how to enable detailed API logging to debug issues like idle drift, tool calling failures, and format mismatches.

## Enable Debug Logging

Set the `DEBUG_API_CALLS` environment variable:

```bash
# In your .env file or export before running
export DEBUG_API_CALLS=1

# Then run experiment
cd ~/CoffeeBench
python -m coffeebench.main --config experiments/roaster_focal_sonnet.toml --seed 0
```

## What Gets Logged

### 1. Raw API Requests
Shows exactly what's being sent to the API:
- Model name
- Messages (conversation history)
- Tools (available functions)
- Tool choice settings
- Max tokens

### 2. Raw API Responses
Shows exactly what the API returns:
- Response ID
- Content blocks (text, tool_use, thinking)
- Tool call IDs and names
- Stop reason
- Token usage

### 3. Message Conversion
Shows how internal messages are converted to Anthropic format:
- Assistant messages with tool_calls
- Tool result messages with tool_call_id
- Detects idle drift patterns

### 4. Parsed Model Response
Shows the final parsed response:
- Whether it has content
- Whether it has tool_calls
- Number of tool calls
- Warnings for idle drift or parallel calls

## Debugging Idle Drift

**Symptoms:**
- Agents repeatedly call `wait_for_next_day` without taking actions
- Agents get stuck in loops
- No economic activity

**What to look for in logs:**

### Pattern 1: Empty Assistant Responses
```
[DEBUG] Parsed ModelResponse:
  - has_content: False
  - has_tool_calls: False
  ⚠️  IDLE DRIFT: No content and no tool_calls!
```

**Cause:** API returned empty response, agent has nothing to do

### Pattern 2: Parallel Tool Calls (Disabled)
```
[DEBUG] Parsed ModelResponse:
  - num_tool_calls: 2
  ⚠️  PARALLEL CALLS: 2 tool calls in one response!
```

**Cause:** API ignored `disable_parallel_tool_use`, agent confused by multiple calls

### Pattern 3: Tool Call ID Mismatch
```
  [5] assistant: tool_call: id=toolu_abc123, name=view_listings
  [6] tool message: tool_call_id=call_xyz789
```

**Cause:** Tool result references wrong tool_call_id, API rejects continuation

## Common Issues with API Gateways

### Issue 1: OpenAI Format vs Anthropic Format

**Problem:** Intenext might translate between OpenAI and Anthropic formats incorrectly

**Check for:**
- Tool call IDs starting with `call_` (OpenAI) vs `toolu_` (Anthropic)
- Different tool_choice formats
- Missing or malformed tool_use blocks

**Example of format mismatch:**
```json
// OpenAI format (wrong for Anthropic)
{
  "role": "assistant",
  "tool_calls": [
    {"id": "call_abc", "function": {"name": "view_listings"}}
  ]
}

// Anthropic format (correct)
{
  "role": "assistant",
  "content": [
    {"type": "tool_use", "id": "toolu_abc", "name": "view_listings"}
  ]
}
```

### Issue 2: Tool Choice Not Respected

**Problem:** API ignores `disable_parallel_tool_use: true`

**Check for:**
- Multiple tool_calls in single response
- Agent receives multiple tool results at once
- Confusion in conversation flow

**Solution:** File issue with Intenext or switch to direct Anthropic API

### Issue 3: Base URL Misconfiguration

**Problem:** Double `/v1/v1/` in URL

**Check for:**
- 404 errors with "Invalid URL" message
- Base URL includes `/v1` suffix

**Solution:** Remove `/v1` from `ANTHROPIC_BASE_URL`:
```bash
# Wrong
ANTHROPIC_BASE_URL="https://api.intenext.ai/v1"

# Correct
ANTHROPIC_BASE_URL="https://api.intenext.ai"
```

## Analyzing Logs

### Step 1: Find Idle Drift Instances

Search logs for warnings:
```bash
grep "IDLE DRIFT" experiment.log
grep "PARALLEL CALLS" experiment.log
```

### Step 2: Check Tool Call ID Matching

For each idle drift, trace back to see if tool_call_id matches:
```bash
# Find assistant message with tool_call
grep -B5 "tool_call: id=" experiment.log

# Find corresponding tool result
grep -A5 "tool message: tool_call_id=" experiment.log
```

### Step 3: Compare Request vs Response

Check if API is returning what you expect:
```bash
# See what was requested
grep -A20 "Anthropic API Request" experiment.log

# See what was returned
grep -A20 "Anthropic API Response" experiment.log
```

## Example Debug Session

```bash
# Enable debug logging
export DEBUG_API_CALLS=1

# Run short experiment (10 days for testing)
python -m coffeebench.main --config experiments/roaster_focal_sonnet.toml --seed 0 --max-days 10 > debug.log 2>&1

# Check for idle drift
grep "IDLE DRIFT" debug.log

# If found, analyze the conversation history
grep -B20 "IDLE DRIFT" debug.log | less

# Check tool_call_id matching
grep "tool_call: id=" debug.log
grep "tool_call_id=" debug.log
```

## Reporting Issues to Intenext

If you find format mismatches or tool calling issues:

1. **Capture the logs** with `DEBUG_API_CALLS=1`
2. **Extract the problematic request/response pair**
3. **Show the mismatch:**
   - What you sent (request)
   - What you got back (response)
   - What you expected
4. **Include the error or idle drift pattern**

Example report:
```
Subject: Tool calling format issue with Claude Sonnet via Intenext

When using Claude Sonnet through Intenext API, I'm seeing idle drift 
where the model returns empty responses.

Request sent:
{
  "model": "claude-sonnet-4-6",
  "tool_choice": {"type": "auto", "disable_parallel_tool_use": true},
  ...
}

Response received:
{
  "content": [],  // Empty!
  "stop_reason": "end_turn"
}

Expected: Tool call or text content

This causes agents to get stuck in idle loops.
```

## Disabling Debug Logging

Debug logging is verbose and slows down experiments. Disable after debugging:

```bash
# Remove from .env or unset
unset DEBUG_API_CALLS

# Or in .env, comment out:
# DEBUG_API_CALLS=1
```

## Performance Impact

- **With DEBUG_API_CALLS=1:** ~10-20% slower due to JSON serialization and printing
- **Without:** Normal speed

Only enable for debugging, not for production runs.
