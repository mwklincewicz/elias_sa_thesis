from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

try:
    import requests
except ImportError:  # pragma: no cover - only used for dependency-light smoke checks.
    requests = None


@dataclass
class JournalPost:
    site: str
    post_url: str
    title: str
    published: str
    text: str
    terms: list[str]


def html_to_text(html: str, max_chars: int) -> str:
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover - exercised only when dependency is missing.
        raise ImportError("beautifulsoup4 is required for live WordPress scraping.") from exc

    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    return text[:max_chars]


def parse_published_at(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def is_within_window(published: str, cutoff_date: datetime | None) -> bool:
    if cutoff_date is None:
        return True
    published_at = parse_published_at(published)
    if published_at is None:
        return False
    return published_at >= cutoff_date


def _site_domain(url: str) -> str:
    return urlparse(url).netloc.lower()


def _collect_term_names_from_wp_rest(post: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    embedded = post.get("_embedded", {})
    for term_group in embedded.get("wp:term", []):
        for term in term_group:
            name = str(term.get("name", "")).strip()
            if name:
                terms.append(name)
    return terms


def _collect_term_names_from_wpcom(post: dict[str, Any]) -> list[str]:
    terms: list[str] = []
    for group_name in ("categories", "tags"):
        group = post.get(group_name, {})
        if isinstance(group, dict):
            for key, value in group.items():
                if isinstance(value, dict):
                    name = str(value.get("name") or value.get("slug") or key).strip()
                else:
                    name = str(key).strip()
                if name:
                    terms.append(name)
    return terms


def _build_post_from_wp_json(site_url: str, post: dict[str, Any], max_chars: int) -> JournalPost:
    rendered = post.get("content", {}).get("rendered", "")
    return JournalPost(
        site=site_url,
        post_url=post.get("link", ""),
        title=post.get("title", {}).get("rendered", ""),
        published=post.get("date", ""),
        text=html_to_text(rendered, max_chars=max_chars),
        terms=_collect_term_names_from_wp_rest(post),
    )


def _build_post_from_reader_api(site_url: str, post: dict[str, Any], max_chars: int) -> JournalPost:
    rendered = post.get("content", "")
    return JournalPost(
        site=site_url,
        post_url=post.get("URL", ""),
        title=post.get("title", ""),
        published=post.get("date", ""),
        text=html_to_text(rendered, max_chars=max_chars),
        terms=_collect_term_names_from_wpcom(post),
    )


def _fetch_from_site_rest_api(
    site_url: str,
    timeout: int,
    max_chars: int,
    limit: int,
    cutoff_date: datetime | None,
) -> list[JournalPost]:
    parsed = urlparse(site_url)
    domain = parsed.netloc or parsed.path
    candidates = [
        f"https://{domain}/wp-json/wp/v2/posts",
        f"http://{domain}/wp-json/wp/v2/posts",
    ]

    last_error: Exception | None = None
    for api_url in candidates:
        try:
            results: list[JournalPost] = []
            page = 1
            while len(results) < limit:
                per_page = min(100, limit - len(results))
                response = requests.get(
                    api_url,
                    params={"per_page": per_page, "page": page, "orderby": "date", "order": "desc", "_embed": 1},
                    timeout=timeout,
                )
                if response.status_code == 400 and page > 1:
                    break
                response.raise_for_status()
                posts: list[dict[str, Any]] = response.json()
                if not posts:
                    break

                stop_due_to_cutoff = False
                for post in posts:
                    journal_post = _build_post_from_wp_json(site_url, post, max_chars)
                    if not is_within_window(journal_post.published, cutoff_date):
                        stop_due_to_cutoff = True
                        break
                    results.append(journal_post)
                    if len(results) >= limit:
                        return results

                if stop_due_to_cutoff:
                    break
                page += 1
            if results:
                return results
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    return []


def _fetch_from_wordpress_reader_tags(
    site_url: str,
    timeout: int,
    max_chars: int,
    limit: int,
    tags: tuple[str, ...],
    cutoff_date: datetime | None,
) -> list[JournalPost]:
    site_domain = _site_domain(site_url)
    results: list[JournalPost] = []
    seen_urls: set[str] = set()

    for tag in tags:
        api_url = f"https://public-api.wordpress.com/rest/v1.1/read/tags/{tag}/posts"
        response = requests.get(api_url, params={"number": 100}, timeout=timeout)
        response.raise_for_status()
        posts: list[dict[str, Any]] = response.json().get("posts", [])
        for post in posts:
            post_url = post.get("URL", "")
            if _site_domain(post_url) != site_domain or post_url in seen_urls:
                continue
            if not is_within_window(post.get("date", ""), cutoff_date):
                continue
            seen_urls.add(post_url)
            results.append(_build_post_from_reader_api(site_url, post, max_chars))
            if len(results) >= limit:
                return results
    return results


def _fetch_from_wordpress_site_api(
    site_url: str,
    timeout: int,
    max_chars: int,
    limit: int,
    cutoff_date: datetime | None,
) -> list[JournalPost]:
    parsed = urlparse(site_url)
    site_id = parsed.netloc or parsed.path
    api_url = f"https://public-api.wordpress.com/rest/v1.1/sites/{site_id}/posts/"

    results: list[JournalPost] = []
    offset = 0

    while len(results) < limit:
        batch_size = min(100, limit - len(results))
        response = requests.get(
            api_url,
            params={"number": batch_size, "offset": offset, "order": "DESC"},
            timeout=timeout,
        )
        response.raise_for_status()
        posts: list[dict[str, Any]] = response.json().get("posts", [])
        if not posts:
            break

        stop_due_to_cutoff = False
        for post in posts:
            journal_post = _build_post_from_reader_api(site_url, post, max_chars)
            if not is_within_window(journal_post.published, cutoff_date):
                stop_due_to_cutoff = True
                break
            results.append(journal_post)
            if len(results) >= limit:
                return results

        if stop_due_to_cutoff or len(posts) < batch_size:
            break
        offset += batch_size

    return results


def fetch_site_posts(
    site_url: str,
    timeout: int,
    max_chars: int,
    limit: int,
    tags: tuple[str, ...] = (),
    cutoff_date: datetime | None = None,
) -> list[JournalPost]:
    if requests is None:
        raise ImportError("requests is required for live WordPress scraping.")

    try:
        posts = _fetch_from_site_rest_api(site_url, timeout, max_chars, limit, cutoff_date)
        if posts:
            return posts
    except Exception:
        pass

    try:
        posts = _fetch_from_wordpress_site_api(site_url, timeout, max_chars, limit, cutoff_date)
        if posts:
            return posts
    except Exception:
        pass

    if tags:
        posts = _fetch_from_wordpress_reader_tags(site_url, timeout, max_chars, limit, tags, cutoff_date)
        if posts:
            return posts

    raise RuntimeError(f"Could not fetch posts for {site_url} via site REST API or WordPress Reader fallback.")
