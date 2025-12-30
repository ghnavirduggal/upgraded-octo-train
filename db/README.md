### Database adapter overview

- Set `CAP_DB_URL` to point to your production database. Leave it unset to use the default `capability.sqlite3`.
- Supported schemes today: `sqlite:///absolute/path/to/file.sqlite3`. To support Postgres/Redshift/etc., add a new adapter class in `db/adapters.py` (following the `SQLiteAdapter` contract) and implement `schema_for_backend()` in `db/schema.py` for that backend.
- Every adapter must expose:
  - `driver`: short name (e.g., `"postgres"`)
  - `connect()`: returns a DB-API compatible connection; all existing modules reuse this API.
  - `info()`: returns metadata (used for logging/diagnostics).
  - Optional: `row_type` for row-type detection.
- Steps to plug in a new backend:
  1. Implement `class PostgresAdapter(DatabaseAdapter)` in `db/adapters.py` (or another module you import there) and extend `configure_adapter()` to instantiate it when `CAP_DB_URL` begins with `postgres://` (or any driver you need).
  2. Update `db/schema.py` with SQL for that backend so `cap_db.init_db()` can bootstrap required tables.
  3. Set `CAP_DB_URL` (e.g., `CAP_DB_URL=postgresql://user:pass@host/dbname`) in your production environment before launching the app. No other code changes are required because the rest of the app opens connections via the adapter.
