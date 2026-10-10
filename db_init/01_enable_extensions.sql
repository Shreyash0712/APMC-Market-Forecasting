-- Enable TimescaleDB (time-series capabilities)
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Enable PostGIS (spatial queries for market proximity)
CREATE EXTENSION IF NOT EXISTS postgis;

-- Enable pg_stat_statements (for query performance tuning and monitoring)
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Verify extensions
SELECT extname, extversion FROM pg_extension;
