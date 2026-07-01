#!/usr/bin/env python3
"""Generate an image via the Databricks AI Gateway (OpenAI-compatible responses API).

Usage:
  DATABRICKS_TOKEN=dapi... uv run --with openai python scripts/generate_image.py \
      "a gray tabby cat hugging an otter with an orange scarf" [out.png]
"""
import base64
import os
import sys

from openai import OpenAI

BASE_URL = "https://eng-ml-inference.staging.cloud.databricks.com/ai-gateway/openai/v1"
MODEL = "databricks-gpt-5-5"


def main() -> int:
    token = os.environ.get("DATABRICKS_TOKEN")
    if not token:
        print("error: set DATABRICKS_TOKEN", file=sys.stderr)
        return 1

    prompt = sys.argv[1] if len(sys.argv) > 1 else (
        "a gray tabby cat hugging an otter with an orange scarf"
    )
    out_path = sys.argv[2] if len(sys.argv) > 2 else "image.png"

    client = OpenAI(api_key=token, base_url=BASE_URL)
    response = client.responses.create(
        model=MODEL,
        input=prompt,
        tools=[{"type": "image_generation"}],
    )

    images = [o.result for o in response.output if o.type == "image_generation_call"]
    if not images:
        print("error: no image returned. Raw output:", file=sys.stderr)
        print(response.output, file=sys.stderr)
        return 2

    with open(out_path, "wb") as f:
        f.write(base64.b64decode(images[0]))
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
