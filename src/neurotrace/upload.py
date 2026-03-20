"""Upload HTML reports to CarbonFiles."""

from __future__ import annotations

import os

import click


def upload_report(html_path: str, bucket_id: str | None = None) -> str:
    """Upload HTML report to CarbonFiles. Returns public URL."""
    try:
        from carbonfiles import CarbonFiles
    except ImportError:
        raise click.ClickException(
            "Upload requires the CarbonFiles SDK: "
            "pip install 'neurotrace[upload]'"
        )

    cf_url = os.environ.get("CF_URL")
    cf_api_key = os.environ.get("CF_API_KEY")

    if not cf_url or not cf_api_key:
        raise click.ClickException(
            "--upload requires CF_URL and CF_API_KEY environment variables."
        )

    cf = CarbonFiles(base_url=cf_url, api_key=cf_api_key)

    if bucket_id is None:
        bucket = cf.buckets.create(
            name="neurotrace-report", expires="7d"
        )
        bucket_id = bucket.id

    cf.buckets[bucket_id].files.upload(html_path)

    filename = os.path.basename(html_path)
    return f"{cf_url}/api/buckets/{bucket_id}/files/{filename}/content"
