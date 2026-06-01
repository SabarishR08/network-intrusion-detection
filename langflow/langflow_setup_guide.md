# LangFlow Setup Guide

## Suggested Flow

1. Add a file loader or CSV input node.
2. Add the detection model node or API call node.
3. Pass the result to an IBM Granite explanation node.
4. Add a formatter node for the final analyst response.

## Recommended Prompt for Granite

"You are a cybersecurity assistant. Explain the intrusion label, likely traffic pattern, and recommended next action in a concise analyst-friendly format."

## What to Capture for Submission

- LangFlow canvas screenshot
- Node labels and connections
- Final output example
- Any Granite prompt configuration used in the demo
