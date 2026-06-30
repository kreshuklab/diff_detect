# Difference Detective

App for annotating visual differences between images, and rating annotation choices.

## Setup

1. Install dependencies:

   ```sh
   pip install -e .
   ```

2. Create a Supabase project and run `schema/supabase.sql` in the Supabase SQL editor.

3. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill in:

   ```toml
   SUPABASE_URL = "https://your-project.supabase.co"
   SUPABASE_KEY = "your-supabase-anon-key"
   ```

4. Start the app:

   ```sh
   streamlit run app.py
   ```

5. Check that the remote Supabase schema is reachable from the app credentials:

   ```sh
   python scripts/check_supabase_schema.py
   ```

`supabase-py` is enough for app reads and writes, but it cannot create tables with an anon key. Creating or resetting tables requires the Supabase SQL editor or a direct Postgres/admin connection.

## Image Data

Each dataset lives in `data/<dataset_id>/` and must include `rounds.json`.
`seeded_annotations.json` is optional. The app plays through every round in
the selected dataset. After login, users choose a dataset from a progress page
that shows completed selections and ratings for each available dataset.

Regenerate them with:

### hf_heliconius

```sh
python scripts/generate_rounds_hf_heliconius_.py --rounds 12 --seed 20260626
```

The generator uses the full Heliconius CSV manifest and filters to non-hybrid, non-duplicate JPG/PNG rows. It includes the available strict same-subspecies example, then fills the remaining rounds with same-view, same-mimic-group tasks where three references share species and subspecies and the odd image uses a different species.

The dataset has only one strict non-hybrid cross-species subspecies candidate with enough images (`ssp.nov.P`, ventral), so most generated rounds use the relaxed same-mimic-group rule.

Each image entry has an `image_id`, local cache `path`, `source_url`, taxonomy metadata, and `species_role`. The app downloads source images into `data/hf_heliconius/images/` on demand and falls back to deterministic placeholders if a URL cannot be reached.

