# Web Scraping Utilities

This repository contains several standalone scripts used to scrape listing data from
popular vacation rental platforms. Each script performs map-based queries over
predefined bounding boxes and writes the results to CSV files. Cookies and tokens
have been obtained manually and are included directly in the code, so they might
expire over time.

## Scripts

- **`web_scraping.py`** – Scrapes Airbnb using Selenium and the Airbnb API.
- **`booking3.py`** – Collects listings from Booking.com via their GraphQL API and optional page crawling.
- **`holidu_scraping2.py`** – Retrieves listings from Holidu.
- **`vrbo_scraping2.py`** – Gathers results from VRBO with the help of Selenium.

Each script defines an initial bounding box for the Mallorca area and recursively
subdivides that region when the number of listings is too large. Data such as
licence numbers, host names and geolocation details are extracted and written to
CSV files in the current directory.

## Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the desired script. The scripts can be executed independently; for example:

```bash
python booking3.py
```

3. Output CSV files will be created in the repository root.

Some scripts rely on cookies or tokens that might need to be refreshed. Update
those values in the source code if requests start failing.

## Disclaimer

These scripts are for educational purposes. Make sure scraping is allowed by the
terms of service of each website before running them.
