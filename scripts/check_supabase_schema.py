from __future__ import annotations

from pathlib import Path

import toml

from supabase import create_client

REQUIRED_TABLES = ("users", "submissions", "ratings")
REQUIRED_SELECTS = {
    "users": "username,password",
    "submissions": (
        "username,dataset_id,task_id,selected_image_id,labels,explanation,"
        "canvas_json,annotation_layers,composite_png_base64,created_at"
    ),
    "ratings": (
        "username,dataset_id,task_id,winner_source,winner_submission_id,"
        "option_payload,created_at"
    ),
}


def main() -> None:
    secrets_path = Path(".streamlit/secrets.toml")
    if not secrets_path.exists():
        raise SystemExit("Missing .streamlit/secrets.toml")

    secrets = toml.loads(secrets_path.read_text(encoding="utf-8"))
    url = secrets.get("SUPABASE_URL")
    key = secrets.get("SUPABASE_KEY")
    if not url or not key:
        raise SystemExit(
            "SUPABASE_URL and SUPABASE_KEY are required in .streamlit/secrets.toml"
        )

    client = create_client(url, key)
    missing = []
    for table_name in REQUIRED_TABLES:
        try:
            client.table(table_name).select(REQUIRED_SELECTS[table_name]).limit(
                1
            ).execute()
        except Exception as exc:
            missing.append((table_name, str(exc)))

    if missing:
        print("Missing or outdated Supabase schema:")
        for table_name, error in missing:
            print(f"- public.{table_name}: {error}")
        print(
            "Run schema/supabase.sql in the Supabase SQL editor."
        )
        raise SystemExit(1)

    print(
        "Supabase schema check passed: users, submissions, and ratings have the required columns."
    )


if __name__ == "__main__":
    main()
