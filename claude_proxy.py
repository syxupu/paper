#!/usr/bin/env python3
"""Minimal OpenAI-compatible proxy that routes to `claude -p`."""
import http.server
import json
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path

PORT = 47892

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silence access logs

    def do_POST(self):
        if self.path not in ("/v1/chat/completions", "/chat/completions"):
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        messages = body.get("messages", [])

        # Flatten messages into a single prompt.
        # NOTE: Do NOT use --system-prompt flag — it triggers a different API
        # path that the proxy rejects. Instead, prepend system content inline.
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.insert(0, f"[System Instructions]\n{content}")
            elif role == "assistant":
                parts.append(f"[Previous Response]\n{content}")
            else:
                parts.append(content)

        prompt = "\n\n".join(parts) if parts else "Hello"

        cmd = ["claude", "-p", prompt, "--no-session-persistence"]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True, text=True, timeout=1800
            )
            content = result.stdout.strip()
            if result.returncode != 0 and not content:
                content = f"[claude error] {result.stderr.strip()[:500]}"
        except subprocess.TimeoutExpired:
            content = "[timeout]"
        except Exception as e:
            content = f"[error: {e}]"

        resp = {
            "id": f"rc-{int(time.time())}",
            "object": "chat.completion",
            "model": body.get("model", "claude"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        payload = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def run():
    server = http.server.HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[claude_proxy] listening on http://127.0.0.1:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    run()
