"""Tiny CLI client for the Gemini docs MCP (streamable HTTP).

Usage:
  python phase0/docs_mcp.py list
  python phase0/docs_mcp.py search "query" [language] [scope]
  python phase0/docs_mcp.py get "<doc id or url>"
"""
import json
import sys
import urllib.request

URL = "https://gemini-api-docs-mcp.dev"
HDR = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}


def rpc(method, params, _id=1):
    body = json.dumps({"jsonrpc": "2.0", "id": _id, "method": method, "params": params}).encode()
    req = urllib.request.Request(URL, data=body, headers=HDR, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read().decode()
    for line in raw.splitlines():
        if line.startswith("data:"):
            return json.loads(line[5:])
    return json.loads(raw)


def call(tool, args):
    res = rpc("tools/call", {"name": tool, "arguments": args})
    if "error" in res:
        return json.dumps(res["error"], indent=1)
    out = []
    for c in res["result"].get("content", []):
        out.append(c.get("text", json.dumps(c)))
    return "\n".join(out)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "list":
        for t in rpc("tools/list", {})["result"]["tools"]:
            print(t["name"], "::", json.dumps(t["inputSchema"].get("properties", {}))[:600])
            print("   ", t.get("description", "")[:300].replace("\n", " "))
    elif cmd == "search":
        args = {"query": sys.argv[2]}
        if len(sys.argv) > 3 and sys.argv[3]:
            args["language"] = sys.argv[3]
        if len(sys.argv) > 4 and sys.argv[4]:
            args["scope"] = sys.argv[4]
        txt = call("gemini_search_docs", args)
        try:
            for h in json.loads(txt)["hits"]:
                print("*", h["chunk_id"], "\n   ", h["snippet"][:220])
        except Exception:
            print(txt)
    elif cmd == "get":
        print(call("gemini_get_doc", json.loads(sys.argv[2]) if sys.argv[2].startswith("{") else {"chunk_id": sys.argv[2]}))
    elif cmd == "raw":
        print(call(sys.argv[2], json.loads(sys.argv[3])))
