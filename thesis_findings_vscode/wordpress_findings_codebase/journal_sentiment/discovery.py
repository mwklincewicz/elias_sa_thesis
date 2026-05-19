from __future__ import annotations

from urllib.parse import urlparse

import requests


DEFAULT_TAGS = (
    "journal",
    "personal",
    "daily",
    "life",
    "mental-health",
    "diary",
    "travel",
    "thoughts",
)


def clean_domain(url: str | None) -> str | None:
    if not url:
        return None
    try:
        parsed = urlparse(url)
        if not parsed.netloc:
            return None
        return f"{parsed.scheme or 'https'}://{parsed.netloc}"
    except ValueError:
        return None


def discover_wordpress_sites(tags: tuple[str, ...], timeout: int) -> list[str]:
    found_sites: set[str] = set()
    for tag in tags:
        url = f"https://public-api.wordpress.com/rest/v1.1/read/tags/{tag}/posts"
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        for post in payload.get("posts", []):
            domain = clean_domain(post.get("URL"))
            if domain:
                found_sites.add(domain)
    return sorted(found_sites)
