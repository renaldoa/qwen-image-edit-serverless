#!/usr/bin/env python3
"""Deploy Qwen-Image-Edit-2511 on a RunPod A40 on-demand pod."""

import requests
import time
import json
import sys
import os

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
if not RUNPOD_API_KEY:
    print("Set RUNPOD_API_KEY environment variable.")
    sys.exit(1)
GRAPHQL_URL = f"https://api.runpod.io/graphql?api_key={RUNPOD_API_KEY}"


def graphql(query):
    resp = requests.post(GRAPHQL_URL, json={"query": query})
    data = resp.json()
    if "errors" in data:
        print(f"GraphQL errors:\n{json.dumps(data['errors'], indent=2)}")
        sys.exit(1)
    return data["data"]


def create_pod():
    query = """
    mutation {
      podFindAndDeployOnDemand(input: {
        cloudType: ALL
        gpuCount: 1
        volumeInGb: 100
        containerDiskInGb: 20
        gpuTypeId: "NVIDIA A40"
        name: "qwen-image-edit"
        imageName: "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"
        ports: "7860/http,22/tcp"
        volumeMountPath: "/workspace"
        env: [
          { key: "GRADIO_SERVER_NAME", value: "0.0.0.0" }
          { key: "HF_HOME", value: "/workspace/hf_cache" }
        ]
      }) {
        id
        desiredStatus
        imageName
      }
    }
    """
    data = graphql(query)
    pod = data["podFindAndDeployOnDemand"]
    return pod["id"]


def get_pod(pod_id):
    query = f"""
    query {{
      pod(input: {{ podId: "{pod_id}" }}) {{
        id
        name
        desiredStatus
        runtime {{
          uptimeInSeconds
          ports {{
            ip
            isIpPublic
            privatePort
            publicPort
            type
          }}
        }}
      }}
    }}
    """
    data = graphql(query)
    return data["pod"]


def wait_for_pod(pod_id, timeout=300):
    print("Waiting for pod to start", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        pod = get_pod(pod_id)
        runtime = pod.get("runtime")
        if runtime and runtime.get("uptimeInSeconds", 0) > 0:
            print("\nPod is running!")
            return pod
        print(".", end="", flush=True)
        time.sleep(5)
    print("\nTimed out waiting for pod.")
    sys.exit(1)


def generate_setup_command():
    """Read app.py and produce a paste-able setup command."""
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    if not os.path.exists(app_path):
        return None
    with open(app_path) as f:
        app_code = f.read()

    lines = [
        "cat > /workspace/app.py << 'APPEOF'",
        app_code.rstrip(),
        "APPEOF",
        "",
        "pip install gradio accelerate 'transformers>=4.51.3' \\",
        "  'git+https://github.com/huggingface/diffusers' && \\",
        "cd /workspace && python app.py",
    ]
    return "\n".join(lines)


def main():
    print("Creating RunPod A40 on-demand pod for Qwen-Image-Edit-2511 ...")
    pod_id = create_pod()
    print(f"Pod created: {pod_id}")

    pod = wait_for_pod(pod_id)

    gradio_url = f"https://{pod_id}-7860.proxy.runpod.net"
    console_url = "https://www.runpod.io/console/pods"

    print()
    print("=" * 62)
    print(f"  Pod ID       : {pod_id}")
    print(f"  Gradio URL   : {gradio_url}")
    print(f"  RunPod Console: {console_url}")
    print("=" * 62)
    print()
    print("NEXT STEPS")
    print("----------")
    print("1. Open the RunPod console link above")
    print("2. Click 'Connect' on your pod -> 'Start Web Terminal'")
    print("3. Paste the command block below into the terminal")
    print("4. Wait for dependencies + model download (~10-15 min first time)")
    print(f"5. Open {gradio_url}")
    print()

    cmd = generate_setup_command()
    if cmd:
        print("--- PASTE THIS INTO THE WEB TERMINAL ---")
        print()
        print(cmd)
        print()
        print("--- END ---")
    else:
        print("Could not find app.py next to this script.")
        print("Copy app.py to the pod manually and run:")
        print("  pip install gradio accelerate 'transformers>=4.51.3' \\")
        print("    'git+https://github.com/huggingface/diffusers'")
        print("  cd /workspace && python app.py")


if __name__ == "__main__":
    main()
