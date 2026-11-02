# Security Policy

## Reporting a Vulnerability

Please do not report security vulnerabilities in public GitHub issues.

If you discover a security bug in `quantpreflight` (for example, an unbounded allocation when parsing malformed Safetensors or GGUF headers, path traversal when resolving shard filenames, or credential leakage on cross-origin HTTP redirects), report it privately through GitHub Security Advisories at:

https://github.com/layer8labs8/quantpreflight/security/advisories/new

Include a minimal reproduction (such as a synthetic header fixture) and the affected version. We aim to acknowledge reports within 72 hours.

## Security Design Guarantees

- `quantpreflight` never unpickles `.bin` or `.pt` PyTorch archives and never executes checkpoint code (`trust_remote_code` is never evaluated).
- Safetensors and GGUF binary parsers enforce strict header byte ceilings (`64 MiB` max Safetensors JSON header, bounded GGUF metadata counts and string lengths) before allocating memory.
- Shard paths inside `model.safetensors.index.json` are confined to the checkpoint root; `..` traversal, absolute paths, and URL-scheme overrides inside shard entries are rejected.
- `HF_TOKEN` / `Authorization` headers are stripped automatically whenever an HTTPS `302`/`307` redirect crosses to a different host (such as CDN blob storage).
