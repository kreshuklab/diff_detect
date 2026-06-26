alter table if exists public.submissions
  add column if not exists labels jsonb,
  add column if not exists annotation_layers jsonb;
