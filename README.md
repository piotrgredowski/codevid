# Codevid

Generate video tutorials from automated tests using LLMs.

## Installation

```bash
pip install codevid
```

## Usage

```bash
# Generate a tutorial from a Playwright test
codevid generate tests/test_login.py -o tutorial.mp4

# Preview the script without recording
codevid preview tests/test_login.py

# Initialize a new project
codevid init
```

## Configuration

Create a `codevid.yaml` file:

```yaml
llm:
  provider: anthropic

tts:
  provider: edge
  voice: en-US-AriaNeural

video:
  include_captions: true
```
