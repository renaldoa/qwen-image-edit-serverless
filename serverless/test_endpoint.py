#!/usr/bin/env python3
"""Test a RunPod serverless endpoint for Qwen-Image-Edit-2511."""

import argparse
import base64
import json
import os
import sys
import time

import requests

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
if not RUNPOD_API_KEY:
    print("Set RUNPOD_API_KEY environment variable.")
    sys.exit(1)


def load_image_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def submit_job(endpoint_id, payload):
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()


def poll_status(endpoint_id, job_id, timeout=600):
    url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    start = time.time()
    print("Waiting for result", end="", flush=True)
    while time.time() - start < timeout:
        resp = requests.get(url, headers=headers)
        data = resp.json()
        status = data.get("status")
        if status == "COMPLETED":
            print(" done!")
            return data["output"]
        if status == "FAILED":
            print(" FAILED")
            print(json.dumps(data, indent=2))
            sys.exit(1)
        print(".", end="", flush=True)
        time.sleep(3)
    print("\nTimed out.")
    sys.exit(1)


def save_result(output, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    if "image" in output:
        path = os.path.join(output_dir, "result.png")
        with open(path, "wb") as f:
            f.write(base64.b64decode(output["image"]))
        print(f"Saved: {path}")
    elif "images" in output:
        for i, b64 in enumerate(output["images"]):
            path = os.path.join(output_dir, f"result_{i}.png")
            with open(path, "wb") as f:
                f.write(base64.b64decode(b64))
            print(f"Saved: {path}")
    elif "error" in output:
        print(f"Error from worker: {output['error']}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Test Qwen Image Edit serverless endpoint")
    parser.add_argument("endpoint_id", help="RunPod endpoint ID")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("prompt", help="Edit prompt")
    parser.add_argument("--output-dir", default="./output", help="Directory to save results")
    parser.add_argument("--steps", type=int, default=40, help="Inference steps")
    parser.add_argument("--cfg-scale", type=float, default=4.0, help="True CFG scale")
    parser.add_argument("--seed", type=int, default=-1, help="Seed (-1 = random)")
    parser.add_argument("--num-images", type=int, default=1, help="Number of images to generate")
    args = parser.parse_args()

    print(f"Loading image: {args.image}")
    b64_image = load_image_b64(args.image)
    print(f"  Size: {len(b64_image)} chars base64")

    payload = {
        "input": {
            "image": b64_image,
            "prompt": args.prompt,
            "num_inference_steps": args.steps,
            "true_cfg_scale": args.cfg_scale,
            "seed": args.seed,
            "num_images_per_prompt": args.num_images,
        }
    }

    print(f"\nSubmitting job to endpoint {args.endpoint_id} ...")
    result = submit_job(args.endpoint_id, payload)
    job_id = result["id"]
    print(f"Job ID: {job_id}")

    output = poll_status(args.endpoint_id, job_id)
    save_result(output, args.output_dir)
    print("\nDone!")


if __name__ == "__main__":
    main()
