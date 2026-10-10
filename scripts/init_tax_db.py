import os
import psycopg2
from dotenv import load_dotenv

def init_taxes():
    load_dotenv()
    DB_URL = os.environ.get("DB_URL")
    
    query = """
    CREATE TABLE IF NOT EXISTS state_apmc_taxes (
        state_name VARCHAR(100) PRIMARY KEY,
        tax_rate_decimal NUMERIC(5, 4) NOT NULL
    );

    INSERT INTO state_apmc_taxes (state_name, tax_rate_decimal) VALUES
    ('Punjab', 0.0850),
    ('Haryana', 0.0650),
    ('Madhya Pradesh', 0.0350),
    ('Maharashtra', 0.0200),
    ('Gujarat', 0.0150),
    ('DEFAULT', 0.0300)
    ON CONFLICT (state_name) DO UPDATE SET tax_rate_decimal = EXCLUDED.tax_rate_decimal;
    """
    
    conn = psycopg2.connect(DB_URL)
    cursor = conn.cursor()
    cursor.execute(query)
    conn.commit()
    cursor.close()
    conn.close()
    print("Tax table successfully created and populated in TimescaleDB!")

if __name__ == "__main__":
    init_taxes()
