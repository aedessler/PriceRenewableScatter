# Renewable Energy vs Wholesale Price Analysis

This project analyzes the relationship between renewable energy share and wholesale electricity prices using two different datasets and approaches: European country-level data and US Regional Transmission Organization (RTO) data.

## Files Overview

### 1. `plotPrice.py`
Analyzes European wholesale electricity prices vs renewable energy share for 2022 and 2023.

**Data Sources:**
- `data/european_wholesale_electricity_price_data_monthly.csv` - Monthly wholesale electricity prices by European country (2015-present)
  - Source: [Ember Energy European Wholesale Electricity Price Data](https://ember-energy.org/data/european-wholesale-electricity-price-data/)
  - Direct download: https://storage.googleapis.com/emb-prod-bkt-publicdata/public-downloads/price/outputs/european_wholesale_electricity_price_data_monthly.csv
  - Format: Country, ISO3 Code, Date, Price (EUR/MWh)
- `data/share-of-electricity-production-from-solar-and-wind/` - Renewable energy share data
  - Main file: `share-of-electricity-production-from-solar-and-wind.csv`
  - Source: [Our World in Data - Share of electricity production from solar and wind](https://ourworldindata.org/grapher/share-of-electricity-production-from-solar-and-wind)
  - Data compiled from Ember and Energy Institute sources
  - Format: Entity, Code, Year, Solar and wind - % electricity
  - Coverage: 1985-2024

**Features:**
- Scatter plots with country labels for both 2022 and 2023
- Least-squares regression line with correlation coefficient (R)
- Formatted axes with percentage and currency symbols (EUR/MWh)
- Consistent styling with seaborn theme
- Professional annotation with country names
- Statistical analysis using scipy for regression

**Generated Plots:**
- **Wholesale Price vs. Renewable Share**: Shows relationship between average wholesale electricity prices (EUR/MWh) and renewable energy percentage by country
- **Retail Price vs. Renewable Share**: Analyzes retail electricity prices (EUR/kWh) from Eurostat data against renewable energy share
- **Taxes + Distribution vs. Renewable Share**: Displays the difference between retail and wholesale prices (representing taxes, distribution costs, and margins) versus renewable energy penetration

**Usage:**
```bash
python plotPrice.py
```

**Note**: All data files are now located in the `data/` directory.

### 2. `renewable_price_analysis.py`
Comprehensive analysis of US RTO data comparing renewable energy percentage (wind + solar) vs load-weighted average wholesale electricity prices for 2023 and 2024.

**Data Sources:**
- Multiple CSV files for different RTOs: MISO, ERCOT, CAISO, PJM, SPP, ISONE, NYISO
- Hourly demand and price data with fuel mix information
- Files located at: `/Users/adessler/Library/Mobile Documents/iCloud~md~obsidian/Documents/BulletVault/plotting code/renewablesSaveMoney/rto_data/`
- Data format: timestamp, load, price, fuel_mix columns for each energy source

**Features:**
- Load-weighted average price calculations for accurate representation
- Renewable percentage calculation (wind + solar as % of total load)
- Single panel plot for 2024 data with trend line
- Summary statistics and correlation analysis
- Professional styling with seaborn theme
- Comprehensive data validation and error handling
- RTO-specific fuel mix handling (some RTOs have different available renewable sources)

**Usage:**
```bash
python renewable_price_analysis.py
```

## Data Sources and Credits

### European Data
- **Price Data**: Ember Energy European Wholesale Electricity Price Data
- **Renewable Data**: Our World in Data compilation from Ember and Energy Institute sources
- **Natural Gas Price Data**: World Bank Commodity Markets data
  - Source: https://thedocs.worldbank.org/en/doc/5d903e848db1d1b83e0ec8f744e55570-0350012021/related/CMO-Historical-Data-Monthly.xlsx
  - File: `data/CMO-Historical-Data-Monthly.xlsx`
- **Electricity Retail Prices**: Eurostat Energy Database (ten00117)
  - Source: https://ec.europa.eu/eurostat/databrowser/view/ten00117/default/table?lang=en
  - File: `data/ten00117_page_spreadsheet.xlsx`
  - Description: Comprehensive European energy statistics including production, consumption, and transformation data
  - Data Type: Annual energy balances and quantities covering electricity, fossil fuels, and renewable energy sources
  - Coverage: EU Member States, EFTA countries, and candidate countries
  - Collection: Part of Eurostat's harmonized energy statistics under Regulation (EC) No 1099/2008
- **Electricity Price Components**: Eurostat Energy Database (nrg_pc_204_c)
  - Source: https://ec.europa.eu/eurostat/databrowser/view/nrg_pc_204_c__custom_17509578/default/table?lang=en
  - File: `data/nrg_pc_204_c__custom_17509578_spreadsheet.xlsx`
  - Description: Detailed breakdown of electricity prices for household consumers with components including energy supply, network costs, taxes, and renewable energy levies
  - Data Type: Annual electricity price components (EUR/kWh) for different consumption bands
  - Coverage: 45 European countries/regions including EU27, Euro area, and individual member states
  - Time Period: 2017-2024
  - Special Features: Includes specific renewable energy taxes and allowances, enabling analysis of renewable energy policy impacts on electricity prices
  - Consumption Bands: Two categories - less than 1,000 kWh and 1,000-2,499 kWh annual consumption
  - Components: 15 different price components including renewable taxes, capacity taxes, environmental taxes, and various allowances
- **Time Period**: 2015-present (analysis focuses on 2022-2023)
- **Coverage**: European countries with ISO3 country codes
- **Update Frequency**: Monthly price data, annual renewable percentage data

### US RTO Data
- **Source**: Regional Transmission Organization hourly data
- **Time Period**: 2023-2024 (hourly data)
- **Coverage**: MISO, ERCOT, CAISO, PJM, SPP, ISONE, NYISO
- **Metrics**: Load-weighted average prices, renewable generation percentages
- **Data Resolution**: Hourly timestamps with fuel mix breakdowns

## Technical Dependencies

Both scripts require:
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- pathlib (for file handling)
- warnings (for clean output)

## Analysis Methodology

The analysis compares renewable energy penetration (percentage of electricity from wind and solar) against wholesale electricity prices to identify potential correlations. The European analysis uses country-level aggregated data, while the US RTO analysis uses high-resolution hourly data with load-weighted price calculations for more accurate market representation.
