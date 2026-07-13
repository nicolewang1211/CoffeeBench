# Using Intenext API with CoffeeBench

This guide explains how to use the [Intenext API](https://api.intenext.ai) gateway to run CoffeeBench with GPT and Claude models.

## What is Intenext?

Intenext is an OpenAI-compatible API gateway that provides access to multiple LLM providers (OpenAI, Anthropic, etc.) through a unified interface.

## Setup Instructions

### 1. Get Your Intenext API Key

Sign up at [https://api.intenext.ai](https://api.intenext.ai) and obtain your API key.

### 2. Configure Environment Variables

Add these to your `.env` file:

```bash
# Intenext API key (used for both OpenAI and Anthropic models)
OPENAI_API_KEY="your-intenext-api-key"
ANTHROPIC_API_KEY="your-intenext-api-key"  # Same key for both

# Intenext base URLs
OPENAI_BASE_URL="https://api.intenext.ai/v1"
ANTHROPIC_BASE_URL="https://api.intenext.ai/v1"  # If Intenext supports Claude
```

### 3. Configure Model Names

Check Intenext's documentation for the exact model names they support. You may need to update your experiment configs with their model identifiers.

For example, if Intenext uses different model names:
```toml
[models]
default = "gpt-4o"  # Or whatever Intenext calls it
roaster_A = "claude-sonnet-4-6"  # Or whatever Intenext calls it
```

### 4. Run Your Experiment

```bash
# Run with Intenext API
python -m coffeebench.main --config experiments/roaster_focal_gpt.toml
```

## Pricing

Check Intenext's pricing page for their rates. They may differ from direct OpenAI/Anthropic pricing.

## Troubleshooting

### Authentication Errors

If you get authentication errors:
- Verify your API key is correct
- Check that `OPENAI_BASE_URL` is set correctly
- Ensure the base URL doesn't have a trailing slash

### Model Not Found Errors

If you get "model not found" errors:
- Check Intenext's documentation for supported model names
- Their model names might differ from OpenAI/Anthropic's official names
- Update your experiment configs with the correct model identifiers

### Rate Limits

Intenext may have different rate limits than direct API access. If you hit rate limits:
- Add delays between requests
- Contact Intenext support for higher limits

## Reverting to Direct API Access

To switch back to direct OpenAI/Anthropic APIs:

1. Remove or comment out the base URL variables in `.env`:
```bash
# OPENAI_BASE_URL="https://api.intenext.ai/v1"
# ANTHROPIC_BASE_URL="https://api.intenext.ai/v1"
```

2. Use your direct API keys:
```bash
OPENAI_API_KEY="sk-..."  # Your OpenAI key
ANTHROPIC_API_KEY="sk-ant-..."  # Your Anthropic key
```

3. Run normally - CoffeeBench will use the default OpenAI/Anthropic endpoints.
