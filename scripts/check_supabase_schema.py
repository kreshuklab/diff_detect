from __future__ import annotations

from pathlib import Path

import toml

from supabase import create_client

REQUIRED_TABLES = (
    "users",
    "images",
    "selection_challenges",
    "selection_tasks",
    "selection_choices",
    "rating_tasks",
    "rating_evals",
)
REQUIRED_SELECTS = {
    "users": "username,password,role",
    "images": "dataset_id,image_id,path,hash_kwargs,image_info,image_group,created_at",
    "selection_challenges": "dataset_id,challenge_id,created_at",
    "selection_tasks": (
        "dataset_id,challenge_id,task_index,images,difficulty,created_at"
    ),
    "selection_choices": (
        "dataset_id,challenge_id,task_index,images,username,user_kind,"
        "choice_index,explanation,annotations,artifacts,created_at"
    ),
    "rating_tasks": "dataset_id,challenge_id,task_index,choices,created_at",
    "rating_evals": (
        "dataset_id,challenge_id,task_index,username,choices,"
        "most_convincing,most_likely_ai,artifacts,created_at"
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

    print("Supabase schema check passed: model tables have the required columns.")


if __name__ == "__main__":
    main()
