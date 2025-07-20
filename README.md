# Renewable Energy vs Wholesale Price Analysis

This project analyzes the relationship between renewable energy share and wholesale electricity prices using European country-level data. The analysis includes multiple approaches: correlation studies, regression analysis with natural gas prices, and detailed electricity price component breakdowns.

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

### 3. `electricity_price_components.py`
Creates detailed stacked bar charts showing electricity price components (generation, network costs, and taxes) for European countries using Eurostat data.

**Data Sources:**
- `data/nrg_pc_204_c__custom_17509578_spreadsheet.xlsx` - Eurostat electricity price components data
  - Source: [Eurostat Energy Database (nrg_pc_204_c)](https://ec.europa.eu/eurostat/databrowser/view/nrg_pc_204_c__custom_17509578/default/table?lang=en)
  - Description: Detailed breakdown of electricity prices for household consumers
  - Components: Energy supply, network costs, taxes, fees, levies, VAT, renewable energy taxes, capacity charges, environmental taxes
  - Coverage: 45 European countries/regions including EU27, Euro area, and individual member states
  - Time Period: 2017-2024
  - Consumption Bands: <1,000 kWh and 1,000-2,499 kWh annual consumption

**Features:**
- Configurable analysis by year and consumption band
- Automatic data loading and processing from multiple Excel sheets
- Stacked bar chart visualization showing price component breakdown
- Summary statistics and data tables
- Professional styling with clear component categories
- Error handling and data validation

**Usage:**
```bash
python electricity_price_components.py
```

### 4. `price_vs_renewable_regression.py`
Performs regression analysis of European electricity prices against natural gas prices and renewable energy share to understand price drivers.

**Data Sources:**
- `data/european_wholesale_electricity_price_data_monthly.csv` - Monthly wholesale electricity prices (Ember Energy)
- `data/CMO-Historical-Data-Monthly.xlsx` - Natural gas prices (World Bank Commodity Markets)
- `data/share-of-electricity-production-from-solar-and-wind/` - Renewable energy share (Our World in Data)

**Analysis Models:**
- **Gas-only model**: `Price = a * Gas_Cost + c`
- **Gas + Renewable model**: `Price = a * Gas_Cost + b * Renewable_Fraction + c`

**Configuration Options:**
- `INCLUDE_RENEWABLE`: Toggle between gas-only and gas+renewable regression
- `INCLUDE_2022`: Option to exclude 2022 data from regression (useful for analyzing COVID-19 impact)

**Usage:**
```bash
python price_vs_renewable_regression.py
```

## Data Sources and Credits

### European Data
- **Price Data**: Ember Energy European Wholesale Electricity Price Data
- **Renewable Data**: Our World in Data compilation from Ember and Energy Institute sources
- **Hydroelectric Data**: Our World in Data compilation from Ember and Energy Institute sources
  - Source: [Our World in Data - Share of electricity production from hydropower](https://ourworldindata.org/grapher/share-electricity-hydro)
  - File: `data/share-electricity-hydro/share-electricity-hydro.csv`
  - Coverage: Global hydroelectric generation share data (1985-2024)
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

## Generated Analysis Products

- **electricity_vs_gas_renewable_with_fit.pdf**: Comprehensive analysis document showing the relationship between electricity prices, natural gas costs, and renewable energy penetration

## Technical Dependencies

All scripts require:
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- pathlib (for file handling)
- warnings (for clean output)

Additional dependencies for specific scripts:
- **price_vs_renewable_regression.py**: scikit-learn (LinearRegression, r2_score)

## Analysis Methodology

The project employs multiple analytical approaches:

1. **Correlation Analysis**: Direct correlation between renewable energy penetration and wholesale electricity prices
2. **Regression Modeling**: Multiple regression analysis incorporating natural gas prices as a control variable
3. **Price Component Analysis**: Breakdown of electricity prices into constituent components (generation, network, taxes)
4. **Comparative Analysis**: Cross-country and cross-regional comparisons

The European analysis uses country-level aggregated data, while the US RTO analysis uses high-resolution hourly data with load-weighted price calculations for more accurate market representation. The regression analysis helps isolate the impact of renewable energy on prices by controlling for natural gas price fluctuations, which are a major driver of electricity costs in gas-fired power generation.
