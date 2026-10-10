/*
File: db_init/02_schema.sql
Role: Database Schema Definition
Description: Defines the primary table for APMC market arrivals and converts it into a TimescaleDB Hypertable partitioned by date. Also sets up a PostGIS geometry column and a GiST index for ultra-fast spatial adjacency queries.
Dependencies: 01_enable_extensions.sql (Requires TimescaleDB and PostGIS)
*/

CREATE TABLE IF NOT EXISTS market_arrivals (
    id BIGINT,
    record_date DATE NOT NULL,
    state_name VARCHAR(100),
    district_name VARCHAR(100),
    market_center_name VARCHAR(100),
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    geom GEOMETRY(Point, 4326), -- PostGIS Spatial Column
    commodity_name VARCHAR(100),
    variety VARCHAR(100),
    arrival_quantity DOUBLE PRECISION,
    min_price DOUBLE PRECISION,
    max_price DOUBLE PRECISION,
    modal_price DOUBLE PRECISION
);

-- Convert the standard table into a TimescaleDB Hypertable partitioned on the 'record_date' column.
SELECT create_hypertable('market_arrivals', 'record_date', if_not_exists => TRUE);

-- Create a spatial index for lightning-fast radius/distance calculations
CREATE INDEX IF NOT EXISTS ix_spatial_market_arrivals ON market_arrivals USING GIST (geom);

-- Create a composite index for commodity and date to speed up time-series lookups
CREATE INDEX IF NOT EXISTS ix_commodity_date_market_arrivals ON market_arrivals (commodity_name, record_date DESC);
