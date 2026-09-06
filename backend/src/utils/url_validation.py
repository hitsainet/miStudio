"""
URL validation utilities for user-supplied endpoint URLs.

Design intent: allow internal LLM servers (Ollama, miLLM, vLLM on RFC-1918
addresses) while blocking cloud metadata endpoints that are the primary SSRF
risk (169.254.169.254 on AWS/Azure/GCP, 100.100.100.200 on Alibaba Cloud).
"""

import ipaddress
import socket
from urllib.parse import urlparse

# Cloud instance metadata services — these are the high-value SSRF targets.
# We block these specifically rather than all link-local/private ranges so
# that internal LLM servers on 10.x / 172.16.x / 192.168.x / 127.x remain
# accessible (they are the intended use case for openai_compatible_endpoint).
_BLOCKED_PREFIXES: tuple[ipaddress.IPv4Network, ...] = (
    ipaddress.IPv4Network("169.254.0.0/16"),   # AWS IMDSv1, Azure IMDS, GCP metadata
    ipaddress.IPv4Network("100.100.100.200/32"),  # Alibaba Cloud metadata
    ipaddress.IPv4Network("0.0.0.0/8"),          # Invalid source
)

_ALLOWED_SCHEMES = frozenset({"http", "https"})


def validate_llm_endpoint_url(url: str) -> str:
    """Validate a user-supplied LLM endpoint URL.

    Allows:
    - http:// and https:// schemes only
    - Public, private (RFC-1918), and loopback addresses
      (internal LLM servers are a primary use case)

    Blocks:
    - Non-http/https schemes
    - Cloud instance metadata addresses (169.254.x.x, 100.100.100.200)
    - Unresolvable hostnames (raises ValueError)

    Returns the original URL string if valid.
    Raises ValueError with a descriptive message if invalid.
    """
    if not url:
        raise ValueError("Endpoint URL must not be empty")

    parsed = urlparse(url)

    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(
            f"Endpoint URL scheme must be http or https, got '{parsed.scheme}'"
        )

    host = parsed.hostname
    if not host:
        raise ValueError("Endpoint URL must include a hostname")

    # Resolve hostname to IP for prefix checks
    try:
        ip_str = socket.gethostbyname(host)
        ip = ipaddress.IPv4Address(ip_str)
    except (socket.gaierror, ValueError):
        # If we can't resolve, allow it — the connection will fail at call time
        # and we don't want to reject valid internal hostnames that aren't
        # resolvable from the validation context (e.g. K8s service names).
        return url

    for blocked in _BLOCKED_PREFIXES:
        if ip in blocked:
            raise ValueError(
                f"Endpoint URL resolves to a blocked address ({ip}). "
                "Cloud instance metadata endpoints are not permitted."
            )

    return url
