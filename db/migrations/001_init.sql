-- First database migration for Supabase (PostgreSQL).
-- Mirrors the in-memory store in backend/app/store.py and the tables listed in Report 2.
-- DRAFT: agree on this as a team before running it on the shared project.

create type crop as enum ('tomato', 'apple', 'corn');
create type scan_status as enum ('diagnosed', 'not_sure');
create type user_role as enum ('gardener', 'expert', 'admin');

-- Supabase Auth owns accounts (auth.users). This table adds our own fields.
create table profiles (
  id uuid primary key references auth.users (id) on delete cascade,
  role user_role not null default 'gardener',
  created_at timestamptz not null default now()
);

create table model_versions (
  id text primary key,                    -- e.g. 'effnet-b0-v3'
  hf_repo text not null,
  hf_tag text not null,
  temperature real not null default 1.0,  -- temperature scaling value
  threshold real not null default 0.70,   -- "not sure" below this
  macro_f1 real,
  is_active boolean not null default false,
  created_at timestamptz not null default now()
);
create unique index one_active_model on model_versions (is_active) where is_active;

create table plants (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  name text not null check (char_length(btrim(name)) between 1 and 60),
  crop crop not null,
  created_at timestamptz not null default now()
);
create index plants_user on plants (user_id);

create table scans (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users (id) on delete cascade,
  plant_id uuid references plants (id) on delete set null,
  crop crop not null,
  status scan_status not null,
  photo_paths text[] not null check (cardinality(photo_paths) between 1 and 3), -- paths in the private bucket
  -- Opt-in only, first 3 characters of the postal code, never GPS.
  postal_prefix text check (postal_prefix ~ '^[KLMNP][0-9][A-Z]$'),
  created_at timestamptz not null default now()
);
create index scans_user on scans (user_id, created_at desc);
create index scans_plant on scans (plant_id, created_at desc);

create table predictions (
  id uuid primary key default gen_random_uuid(),
  scan_id uuid not null references scans (id) on delete cascade,
  model_version text not null references model_versions (id),
  rank smallint not null check (rank between 1 and 3),   -- 1 = top candidate
  label text not null,
  confidence real not null check (confidence between 0 and 1),
  unique (scan_id, rank)
);

create table treatments (
  id uuid primary key default gen_random_uuid(),
  label text not null,                      -- disease or pest label
  title text not null,
  detail text not null,
  kind text not null check (kind in ('practice', 'product')),
  source text not null,
  legal_check boolean not null default false,        -- Allowable List / O. Reg. 63/09
  suitability_check boolean not null default false,  -- OMAFRA guidance
  sort_order smallint not null default 0
);
create index treatments_label on treatments (label);

create table feedback (
  id uuid primary key default gen_random_uuid(),
  scan_id uuid not null references scans (id) on delete cascade,
  user_id uuid not null references auth.users (id) on delete cascade,
  was_correct boolean not null,
  correct_label text,
  created_at timestamptz not null default now()
);

create table unsupported_requests (
  id uuid primary key default gen_random_uuid(),
  user_id uuid references auth.users (id) on delete set null,
  requested text not null check (char_length(requested) <= 120),
  created_at timestamptz not null default now()
);

-- Tier 2: low-confidence scans waiting for an expert.
create table review_queue (
  id uuid primary key default gen_random_uuid(),
  scan_id uuid not null unique references scans (id) on delete cascade,
  reviewer_id uuid references auth.users (id),
  expert_label text,
  reviewed_at timestamptz,
  created_at timestamptz not null default now()
);

-- Row-level security: people can only see their own data.
-- The backend uses the service role key, which bypasses these rules.
alter table profiles enable row level security;
alter table plants enable row level security;
alter table scans enable row level security;
alter table predictions enable row level security;
alter table feedback enable row level security;
alter table unsupported_requests enable row level security;
alter table review_queue enable row level security;
alter table treatments enable row level security;
alter table model_versions enable row level security;

create policy own_profile on profiles for select using (id = auth.uid());
create policy own_plants on plants for all using (user_id = auth.uid()) with check (user_id = auth.uid());
create policy own_scans on scans for select using (user_id = auth.uid());
create policy own_predictions on predictions for select
  using (exists (select 1 from scans s where s.id = scan_id and s.user_id = auth.uid()));
create policy own_feedback on feedback for all using (user_id = auth.uid()) with check (user_id = auth.uid());
create policy read_checked_treatments on treatments for select using (legal_check and suitability_check);
