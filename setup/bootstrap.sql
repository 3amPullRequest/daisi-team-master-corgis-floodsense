-- One-time setup for the flood pipeline. Safe to re-run.
CREATE CATALOG IF NOT EXISTS flood_dev;
CREATE CATALOG IF NOT EXISTS flood_prod;

-- After creating the deploy service principal (Step 8a):
GRANT USE CATALOG, CREATE SCHEMA ON CATALOG flood_prod TO `<application-id>`;