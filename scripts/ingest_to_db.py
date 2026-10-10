"""
File: scripts/ingest_to_db.py
Role: Data Ingestion Pipeline
Description: Loads the filtered perishables dataset using memory-safe Pandas casting, and executes a bulk insert into the local TimescaleDB instance. Computes PostGIS geometry points on the fly.
Dependencies: pandas, psycopg2
"""

import pandas as pd
import psycopg2
import psycopg2.extras
import time
import os
from dotenv import load_dotenv

load_dotenv()
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
FILE_PATH = os.path.join(BASE_DIR, "IDP_APMC_NEW", "filtered_perishables.csv")
DB_URL = os.environ.get("DB_URL")


def load_apmc_data():
    print(f"Loading data from {FILE_PATH}...")
    dtypes = {
        'state_name': 'category',
        'district_name': 'category',
        'market_center_name': 'category',
        'commodity_name': 'category',
        'variety': 'category',
        'arrival_quantity': 'float32',
        'min_price': 'float32',
        'max_price': 'float32',
        'modal_price': 'float32',
        'latitude': 'float32',
        'longitude': 'float32'
    }
    df = pd.read_csv(FILE_PATH, dtype=dtypes, parse_dates=['date'])
    # Handle NaN values before insertion
    df = df.where(pd.notnull(df), None)
    return df

def ingest_data():
    df = load_apmc_data()
    print(f"Data loaded into memory: {len(df)} rows.")

    conn = psycopg2.connect(DB_URL)
    cursor = conn.cursor()

    # The SQL query creates the PostGIS geometry directly from lat/lon during insertion
    insert_query = """
        INSERT INTO market_arrivals (
            id, record_date, state_name, district_name, market_center_name, 
            latitude, longitude, geom, commodity_name, variety, 
            arrival_quantity, min_price, max_price, modal_price
        ) VALUES %s
        ON CONFLICT DO NOTHING
    """
    
    print("Preparing bulk insert payload...")
    # Generator to yield tuples for insertion
    def get_tuples():
        for row in df.itertuples(index=False):
            # Extract values, handle potential NaN coordinates for geom creation
            id_val = getattr(row, 'id', None)
            date_val = row.date
            lat = row.latitude
            lon = row.longitude
            
            # PostGIS syntax: ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)
            # However, execute_values escapes inputs, so we can't directly embed ST_MakePoint in the VALUES block via tuple easily.
            # Instead, we will alter the insert_query slightly or just pass longitude and latitude and compute it inside.
            pass

    # Let's adjust the query to cast the coordinates properly
    insert_query = """
        INSERT INTO market_arrivals (
            id, record_date, state_name, district_name, market_center_name, 
            latitude, longitude, geom, commodity_name, variety, 
            arrival_quantity, min_price, max_price, modal_price
        ) VALUES %s
    """
    
    # Actually, execute_values requires a template to apply ST_MakePoint. 
    # Template: ( %s, %s, %s, %s, %s, %s, %s, ST_SetSRID(ST_MakePoint(CAST(%s AS float), CAST(%s AS float)), 4326), %s, %s, %s, %s, %s, %s )
    template = """(
        %s, %s, %s, %s, %s, 
        %s, %s, 
        ST_SetSRID(ST_MakePoint(%s, %s), 4326), 
        %s, %s, %s, %s, %s, %s
    )"""
    
    tuples = []
    for row in df.itertuples(index=False):
        # row.longitude goes twice (one for column, one for MakePoint), same for lat
        # Wait, if longitude is None, ST_MakePoint fails. 
        lon = row.longitude if pd.notnull(row.longitude) else None
        lat = row.latitude if pd.notnull(row.latitude) else None
        
        tuples.append((
            getattr(row, 'id', None),
            row.date,
            row.state_name,
            row.district_name,
            row.market_center_name,
            lat,
            lon,
            lon, # For ST_MakePoint X
            lat, # For ST_MakePoint Y
            row.commodity_name,
            row.variety,
            row.arrival_quantity,
            row.min_price,
            row.max_price,
            row.modal_price
        ))

    print("Executing bulk insertion into TimescaleDB...")
    start_time = time.time()
    
    psycopg2.extras.execute_values(
        cursor, insert_query, tuples, template=template, page_size=10000
    )
    
    conn.commit()
    cursor.close()
    conn.close()
    
    end_time = time.time()
    print(f"Ingestion successful! Time elapsed: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    ingest_data()
