"""
File: scripts/filter_dataset.py
Role: Data Preprocessing
Description: Iterates over the massive raw APMC dataset in a memory-safe, line-by-line manner. Filters out irrelevant commodities and extracts only highly volatile, perishable crops (Tomato, Onion, Green Chilli, Potato) to drastically reduce dataset size for ST-GNN training.
Dependencies: csv, os
"""

import csv
import os

# File paths
input_file = r"C:\Users\YASHVI Sharma\OneDrive\Desktop\APMC Market Forecasting\IDP_APMC_NEW\apmc-arrivals-and-prices-new-source.csv"
output_file = r"C:\Users\YASHVI Sharma\OneDrive\Desktop\APMC Market Forecasting\IDP_APMC_NEW\filtered_perishables.csv"

# Target perishable crops that suffer from acute price volatility
TARGET_CROPS = {"tomato", "onion", "green chilli", "potato"}

def process_dataset():
    print(f"Reading from: {input_file}")
    
    if not os.path.exists(input_file):
        print("Input file not found!")
        return

    written_rows = 0
    total_rows = 0
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        
        writer.writeheader()
        
        for row in reader:
            total_rows += 1
            commodity = row.get('commodity_name', '').strip().lower()
            
            if commodity in TARGET_CROPS:
                writer.writerow(row)
                written_rows += 1
                
            if total_rows % 500000 == 0:
                print(f"Processed {total_rows} rows... (Found {written_rows} relevant records)")
                
    print("\n--- Processing Complete ---")
    print(f"Total rows scanned: {total_rows}")
    print(f"Relevant rows saved: {written_rows}")
    print(f"Output saved to: {output_file}")
    
    original_size = os.path.getsize(input_file) / (1024 * 1024)
    new_size = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Size reduction: {original_size:.2f} MB -> {new_size:.2f} MB")

if __name__ == "__main__":
    process_dataset()
