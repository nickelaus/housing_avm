import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from selenium import webdriver
import random
import tkinter as tk
from tkinter import filedialog
import os


def create_driver(headless=True):  # Changed default to True
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
    ]
    
    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
        # Additional headless-specific settings
        options.add_argument("--window-size=1920,1080")
        options.add_argument("--start-maximized")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-software-rasterizer")  # Add this line
        options.add_argument("--disable-features=VizDisplayCompositor")  # Add this line
        
    # Enhanced anti-bot detection bypass
    options.add_argument(f'user-agent={random.choice(user_agents)}')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--no-sandbox')
    options.add_argument("--lang=en-US,en;q=0.9")
    options.add_argument("--disable-infobars")
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--ignore-ssl-errors')
    
    # Add realistic browser profiles
    options.add_argument("--enable-javascript")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-popup-blocking")
    
    # Add Chrome preferences
    prefs = {
        "profile.default_content_setting_values.notifications": 2,
        "profile.default_content_settings.popups": 0,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
        "credentials_enable_service": False
    }
    options.add_experimental_option("prefs", prefs)
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    
    return webdriver.Chrome(options=options)

def analyze_remarks_simple(remarks_text):
    remarks_lower = remarks_text.lower()
    renovation_keywords = [
        'renovated', 'remodeled', 'updated', 'updates', 'modernized', 'improvement', 'improvements'
        
        'new kitchen', 'new bath','new bathroom', 'new appliances',
        'new roof', 'new windows', 'new flooring', 'new carpet', 'new carpeting'
        'fresh paint', 'newly painted', 'new cabinets', 'new countertops',
        'new plumbing', 'new electrical', 'new interior'
        
        'total renovation', 'revamped'
    ]
    
    is_renovated = any(keyword in remarks_lower for keyword in renovation_keywords)
    
    return {
        "renovated": is_renovated,
        "confidence": 1.0 if is_renovated else 0.0
    }
def cleanup_dataframe(df):
    # Rename the long URL column to just "URL"
    df = df.rename(columns={
        "URL (SEE https://www.redfin.com/buy-a-home/comparative-market-analysis FOR INFO ON PRICING)": "URL"
    })

    drop_cols = ['NEXT OPEN HOUSE START TIME', 'NEXT OPEN HOUSE END TIME', 'FAVOURITE', 'INTERESTED']
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])
            
    # convert sale date to excel-friendly format
    if 'SOLD DATE' in df.columns:
        # Check if 'SOLD DATE' is already in YYYY-MM-DD format
        sample_date = df['SOLD DATE'].dropna().astype(str).iloc[0] if not df['SOLD DATE'].dropna().empty else ''
        already_converted = False
        date_format_used = None
        
        try:
            # Try to parse as YYYY-MM-DD
            pd.to_datetime(sample_date, format='%Y-%m-%d')
            date_format_used = '%Y-%m-%d'
            already_converted = True
        except Exception:
            already_converted = False

        # Try to detect if Excel has converted the date to m/d/yyyy or m/d/yy
        excel_date_formats = ['%m/%d/%Y', '%m/%d/%y']
        for fmt in excel_date_formats:
            try:
                pd.to_datetime(sample_date, format=fmt)
                already_converted = True
                date_format_used = fmt
                break
            except Exception:
                continue

        # Convert dates using the detected format
        if already_converted:
            sold_dates = pd.to_datetime(df['SOLD DATE'], format=date_format_used, errors='coerce')
        else:
            sold_dates = pd.to_datetime(df['SOLD DATE'], format='%B-%d-%Y', errors='coerce')

        # Get the original index of the 'SOLD DATE' column
        sold_date_idx = df.columns.get_loc('SOLD DATE')

        # Create derived columns
        days_since_sold = (pd.Timestamp.now() - sold_dates).dt.days
        month_sold = sold_dates.dt.month

        # Update the DataFrame in the correct order
        df['SOLD DATE'] = sold_dates.dt.strftime('%Y-%m-%d')
        
        # Add the derived columns
        if 'Days_Since_Sold' in df.columns:
            df = df.drop(columns=['Days_Since_Sold'])
        df.insert(sold_date_idx + 1, 'Days_Since_Sold', days_since_sold)

        if 'Month_Sold' in df.columns:
            df = df.drop(columns=['Month_Sold'])
        df.insert(sold_date_idx + 2, 'Month_Sold', month_sold)

    # Calculate age from 'YEAR BUILT'
    if 'YEAR BUILT' in df.columns:
        current_year = pd.Timestamp.now().year
        # Ensure 'YEAR BUILT' is numeric, coercing errors to NaN
        year_built = pd.to_numeric(df['YEAR BUILT'], errors='coerce')
        
        # Calculate age
        age = current_year - year_built
        
        # Get the index of the 'YEAR BUILT' column to insert 'AGE' next to it
        year_built_idx = df.columns.get_loc('YEAR BUILT')
        df.insert(year_built_idx + 1, 'AGE', age)
    
    return df

def get_listing_features(url, driver):
    features = {
        "STORAGE_STRUCTURE": 0,
        "Garage_Spaces": 0,
        "Basement_Level": 0
    }
    try:
        driver.get(url)
        time.sleep(3)  # Wait for page to load
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Extract remarks
        remarks_div = soup.find("div", {"id": "marketing-remarks-scroll"})
        if remarks_div:
            remarks_text = remarks_div.get_text(strip=True)
            features["Remarks"] = remarks_text
            # Choose one of the analysis methods:
            # renovation_info = analyze_remarks_openai(remarks_text)  # Option 1
            renovation_info = analyze_remarks_simple(remarks_text)     # Option 2
            
            features["Recently_Renovated"] = 1 if renovation_info["renovated"] else 0
            # if "renovation_details" in renovation_info:
            #     features["Renovation_Details"] = renovation_info["renovation_details"]
        

        for entry in soup.find_all("span", class_="entryItemContent"):
            # Get the nested span value if it exists
            nested_span = entry.find("span")
            label_text = entry.text.strip().lower()
            value_text = nested_span.text.strip().lower() if nested_span else ""
            print(f"Label: '{label_text}', Value: '{value_text}'")  # Debug

            # Storage Structures
            if "other structures" in label_text:
                if "storage" in value_text:
                    features["STORAGE_STRUCTURE"] = 1
            # Garage Spaces
            if "# of garage spaces" in label_text:
                try:
                    features["Garage_Spaces"] = int(value_text) if value_text and value_text.isdigit() else 0
                except (ValueError, TypeError):
                    features["Garage_Spaces"] = 0
            # Basement Level
            if "basement details" in label_text:
                if "none" in value_text:
                    features["Basement_Level"] = 0
                elif value_text == "partial":
                    features["Basement_Level"] = 1
                elif "unfinished" in value_text or "full" in value_text:
                    features["Basement_Level"] = 2
                elif value_text == "partially finished":
                    features["Basement_Level"] = 3                    
                elif "finished" in value_text:
                    features["Basement_Level"] = 4

    except Exception as e:
        print(f"Error scraping {url}: {e}")
    return features

def scrape_with_retry(url, max_retries=3, base_delay=3):
    for attempt in range(max_retries):
        try:
            driver = create_driver(headless=True)
            features = get_listing_features(url, driver)
            driver.quit()
            return features
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")  # More detailed error logging
            driver.quit()
            if attempt < max_retries - 1:
                sleep_time = base_delay * (1.5 ** attempt) + random.uniform(1, 4)  # Longer delays
                print(f"Waiting {sleep_time:.2f} seconds before retry...")
                time.sleep(sleep_time)
            else:
                print(f"Failed to scrape {url} after {max_retries} attempts: {e}")
                return {"STORAGE_STRUCTURE": 0, "Garage_Spaces": 0, "Basement_Level": 0}




def main():
    # Add command line argument parsing
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--mode', choices=['scrape', 'cleanup'], default='scrape',
    #                    help='Mode to run: scrape for full scraping, cleanup for just cleaning the CSV')
    # args = parser.parse_args()

    # if args.mode == 'cleanup':
    #     # Just run cleanup on existing enriched CSV
    #     df = pd.read_csv('star_valley_hs_area_redfin_sold_enriched.csv')
    #     cleaned_df = cleanup_dataframe(df)
    #     cleaned_df.to_csv('star_valley_hs_area_redfin_sold_enriched_cleaned.csv', index=False)
    #     print("Cleanup complete. Saved to star_valley_hs_area_redfin_sold_enriched_cleaned.csv")
    # else:
    # Load and filter CSV as before
    file = filedialog.askopenfilename(
        title="Select the CSV file exported from Redfin",
        filetypes=[("CSV files", "*.csv")],
        initialdir="./data/"
    )
    if not file:  # This handles both empty string and None (when cancel is pressed)
        print("No file selected. Exiting.")
        return
    df = pd.read_csv(file, low_memory=False)
    required_cols = ['BEDS', 'BATHS', 'PRICE', 'LOT SIZE']
    filtered_df = df.dropna(subset=required_cols)
    for col in required_cols:
        filtered_df = filtered_df[filtered_df[col].astype(str).str.strip() != '']

    # Scrape features for each listing using a single browser instance
    features_list = []
    records = filtered_df.shape[0]
    counter = 0
    for idx, row in filtered_df.iterrows():
        url = row['URL (SEE https://www.redfin.com/buy-a-home/comparative-market-analysis FOR INFO ON PRICING)']
        features = scrape_with_retry(url)
        print(f"Scraped features for {url}: {features}")
        features_list.append(features)
        time.sleep(3 + random.random() * 4)  # Increase delay between requests (3-7 seconds)
        counter += 1
        print(f"Progress: {counter}/{records} ({(counter/records)*100:.2f}%)")
    features_df = pd.DataFrame(features_list)
    result_df = filtered_df.reset_index(drop=True).join(features_df)


    cleaned_result_df = cleanup_dataframe(result_df)
    # Save the updated CSV
    # Get the base filename without extension and add the suffix
    base_name = os.path.splitext(os.path.basename(file))[0]
    output_filename = f"data/{base_name}_enriched_cleaned.csv"
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    cleaned_result_df.to_csv(output_filename, index=False)
    print(f"Saved enriched and cleaned data to: {output_filename}")





if __name__ == "__main__":
    main()

