#!/usr/bin/env python3
"""Deploy a RunPod serverless endpoint for Qwen-Image-Edit-2511.

Uses the REST API to create the endpoint with model caching enabled.
RunPod caches the HF model on their infra so workers load from cache
instead of downloading every cold start.
"""

import json
import os
import sys

import requests

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
if not RUNPOD_API_KEY:
    print("Set RUNPOD_API_KEY environment variable.")
    sys.exit(1)
REST_URL = "https://rest.runpod.io/v1"

IMAGE_NAME = "ghcr.io/renaldoa/qwen-image-edit-serverless:latest"
ENDPOINT_NAME = "qwen-image-edit"
MODEL_NAME = "Qwen/Qwen-Image-Edit-2511"


def rest_api(method, path, payload=None):
    url = f"{REST_URL}{path}"
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }
    resp = requests.request(method, url, headers=headers, json=payload)
    if resp.status_code >= 400:
        print(f"API error ({resp.status_code}):\n{resp.text}")
        sys.exit(1)
    return resp.json()


def create_template():
    """Create a serverless template with the Docker image."""
    payload = {
        "name": f"{ENDPOINT_NAME}-template",
        "imageName": IMAGE_NAME,
        "containerDiskInGb": 20,
        "env": {
            "HF_HOME": "/runpod-volume/huggingface-cache",
        },
    }
    data = rest_api("POST", "/templates", payload)
    return data["id"]


def create_endpoint(template_id):
    """Create a serverless endpoint with model caching."""
    payload = {
        "name": ENDPOINT_NAME,
        "templateId": template_id,
        "gpuTypeIds": ["NVIDIA H100 NVL"],
        "gpuCount": 1,
        "workersMin": 0,
        "workersMax": 3,
        "idleTimeout": 5,
        "scalerType": "QUEUE_DELAY",
        "scalerValue": 4,
        "executionTimeoutMs": 600000,
    }
    return rest_api("POST", "/endpoints", payload)


def main():
    print(f"Creating serverless endpoint '{ENDPOINT_NAME}' ...")
    print(f"  Image: {IMAGE_NAME}")
    print(f"  Model: {MODEL_NAME} (cached by RunPod)")
    print(f"  GPU:   NVIDIA H100 NVL")
    print()

    print("Step 1: Creating template ...")
    template_id = create_template()
    print(f"  Template ID: {template_id}")

    print("Step 2: Creating endpoint ...")
    endpoint = create_endpoint(template_id)
    endpoint_id = endpoint["id"]

    print()
    print("Endpoint created!")
    print()
    print("=" * 62)
    print(f"  Endpoint ID  : {endpoint_id}")
    print(f"  Name         : {endpoint['name']}")
    print(f"  Workers      : {endpoint['workersMin']}-{endpoint['workersMax']}")
    print(f"  Idle timeout : {endpoint['idleTimeout']}s")
    print("=" * 62)
    print()
    print("NOTE: Set the cached model in the RunPod console:")
    print(f"  1. Go to Serverless > {ENDPOINT_NAME} > Edit Endpoint")
    print(f"  2. Set Model field to: {MODEL_NAME}")
    print("  3. Save — RunPod will cache the model on their infra")
    print()
    print("EXAMPLE USAGE")
    print("-------------")
    print()
    print("# Submit a job (async):")
    print(f'curl -s -X POST "https://api.runpod.ai/v2/{endpoint_id}/run" \\')
    print(f'  -H "Authorization: Bearer {RUNPOD_API_KEY}" \\')
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"input": {"image": "<BASE64_IMAGE>", "prompt": "Make the sky purple"}}\'')
    print()
    print("# Check job status:")
    print(f'curl -s "https://api.runpod.ai/v2/{endpoint_id}/status/<JOB_ID>" \\')
    print(f'  -H "Authorization: Bearer {RUNPOD_API_KEY}"')
    print()
    print("# Synchronous (waits for result):")
    print(f'curl -s -X POST "https://api.runpod.ai/v2/{endpoint_id}/runsync" \\')
    print(f'  -H "Authorization: Bearer {RUNPOD_API_KEY}" \\')
    print('  -H "Content-Type: application/json" \\')
    print('  -d \'{"input": {"image": "<BASE64_IMAGE>", "prompt": "Make the sky purple"}}\'')


if __name__ == "__main__":
    main()
