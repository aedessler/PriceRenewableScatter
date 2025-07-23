# Renewable Energy vs Electricity Price Analysis

This project analyzes the relationship between renewable energy share and electricity prices using comprehensive datasets covering European countries and US Regional Transmission Organizations (RTOs). The analysis employs multiple methodologies including correlation studies and regression analysis to understand how renewable energy deployment affects electricity costs.

## Project Overview

The analysis addresses key questions about renewable energy economics:
- **Does higher renewable energy penetration correlate with electricity prices?**
- **What role do natural gas prices play as the primary driver of electricity costs?**
- **What are the differences between wholesale and retail electricity markets?**

## Analysis Scripts

### 1. `plotPrice.py`
**European Price vs Renewable Energy Analysis**

Comprehensive analysis of European wholesale and retail electricity prices versus renewable energy share for multiple years, with focus on 2022-2023 data.

**Key Features:**
- **Wholesale Price Analysis**: Scatter plots of wholesale electricity prices (EUR/MWh) vs renewable share by country
- **Retail Price Analysis**: Analysis of household retail electricity prices (EUR/kWh) vs renewable penetration  
- **Price Component Breakdown**: Examines the difference between retail and wholesale prices (taxes, distribution, margins)
- **Statistical Analysis**: Least-squares regression with correlation coefficients and trend lines
- **Professional Visualization**: Country-labeled scatter plots with formatted axes and statistical annotations

**Data Sources:**
- European wholesale electricity prices (Ember Energy)
- Renewable energy share data (Our World in Data/Ember)
- Eurostat retail electricity price data

**Generated Outputs:**
- Multiple scatter plot visualizations showing price-renewable relationships
- Statistical correlations and regression analysis results

**Usage:**
```bash
python plotPrice.py
```

### 2. `price_vs_renewable_regression.py`
**Natural Gas Price Regression Analysis**

Comprehensive regression analysis demonstrating natural gas prices as the primary driver of European electricity wholesale prices, with configurable analysis options.

**Regression Model:**
```
Electricity_Price = a × Gas_Price + c
```

**Key Features:**
- **Country-by-Country Analysis**: Individual linear regression for each European country
- **Multi-Page PDF Output**: Comprehensive visual report with individual country plots
- **Statistical Rigor**: R² values, correlation coefficients, and regression statistics
- **Configurable Analysis**: Option to include/exclude specific years (e.g., 2022 energy crisis)
- **Data Point Labeling**: Year-by-year visualization on scatter plots
- **Summary Statistics**: Average coefficients across countries and model performance

**Analysis Options:**
- `INCLUDE_2022`: Toggle to exclude 2022 data from regression fitting while retaining for visualization
- Handles missing data and validates sufficient data points for regression

**Generated Outputs:**
- `electricity_vs_gas_renewable_with_fit.pdf` - Multi-page analysis report
- Console output with detailed regression statistics
- Summary plots showing coefficient distributions

**Usage:**
```bash
python price_vs_renewable_regression.py
```

### 3. `renewable_scatter_US.py`
**US Regional Transmission Organization (RTO) Analysis**

Analysis of renewable energy penetration vs load-weighted wholesale electricity prices across major US electricity markets using high-resolution hourly data.

**Key Features:**
- **Load-Weighted Analysis**: Accurate price calculations weighted by actual electricity demand
- **Multi-RTO Coverage**: MISO, ERCOT, CAISO, PJM, SPP, ISONE, NYISO
- **Renewable Calculation**: Wind + solar as percentage of total load
- **Hourly Data Processing**: High-resolution temporal analysis (8,760+ hours per year)
- **Comparative Visualization**: 2023 vs 2024 analysis with trend lines
- **Statistical Analysis**: Correlation analysis and trend identification

**RTO Coverage:**
- **MISO**: Midcontinent Independent System Operator
- **ERCOT**: Electric Reliability Council of Texas  
- **CAISO**: California Independent System Operator
- **PJM**: PJM Interconnection (Mid-Atlantic/Great Lakes)
- **SPP**: Southwest Power Pool
- **ISONE**: ISO New England
- **NYISO**: New York Independent System Operator

**Data Processing:**
- Hourly demand, price, and fuel mix data processing
- Load-weighted average price calculations for market accuracy
- Renewable percentage calculations handling different RTO data structures
- Data validation and error handling for missing/invalid data

**Generated Outputs:**
- Single-panel 2024 renewable vs price scatter plot with trend analysis
- Summary statistics and correlation analysis
- Comprehensive console output with RTO-by-RTO results

**Usage:**
```bash
python renewable_scatter_US.py
```

## Data Sources and Structure

### European Data Sources

#### Wholesale Electricity Prices
- **Source**: [Ember Energy European Wholesale Electricity Price Data](https://ember-energy.org/data/european-wholesale-electricity-price-data/)
- **File**: `data/european_wholesale_electricity_price_data_monthly.csv`
- **Coverage**: European countries, 2015-present, monthly data
- **Format**: Country, ISO3 Code, Date, Price (EUR/MWh)
- **Direct Download**: https://storage.googleapis.com/emb-prod-bkt-publicdata/public-downloads/price/outputs/european_wholesale_electricity_price_data_monthly.csv

#### Renewable Energy Share Data
- **Source**: [Our World in Data - Share of electricity production from solar and wind](https://ourworldindata.org/grapher/share-of-electricity-production-from-solar-and-wind)
- **File**: `data/share-of-electricity-production-from-solar-and-wind/share-of-electricity-production-from-solar-and-wind.csv`
- **Compilation**: Ember and Energy Institute data processed by Our World in Data
- **Coverage**: Global data, 1985-2024, annual values
- **Format**: Entity, Code, Year, Solar and wind - % electricity
- **Citation**: Ember (2025); Energy Institute - Statistical Review of World Energy (2024) – with major processing by Our World in Data

#### Hydroelectric Generation Data  
- **Source**: [Our World in Data - Share of electricity production from hydropower](https://ourworldindata.org/grapher/share-electricity-hydro)
- **File**: `data/share-electricity-hydro/share-electricity-hydro.csv`
- **Coverage**: Global hydroelectric generation share, 1985-2024
- **Format**: Entity, Code, Year, Hydro - % electricity
- **Citation**: Ember (2025); Energy Institute - Statistical Review of World Energy (2025) – with major processing by Our World in Data

#### Natural Gas Price Data
- **Source**: [World Bank Commodity Markets](https://thedocs.worldbank.org/en/doc/5d903e848db1d1b83e0ec8f744e55570-0350012021/related/CMO-Historical-Data-Monthly.xlsx)
- **File**: `data/CMO-Historical-Data-Monthly.xlsx`
- **Coverage**: Monthly European natural gas prices, historical data
- **Format**: Date, Natural gas Europe ($/MMBtu)
- **Usage**: Primary driver analysis in regression modeling

#### Retail Electricity Prices
- **Source**: [Eurostat Energy Database (ten00117)](https://ec.europa.eu/eurostat/databrowser/view/ten00117/default/table?lang=en)
- **File**: `data/ten00117_page_spreadsheet.xlsx`
- **Description**: Comprehensive European energy statistics
- **Coverage**: EU Member States, EFTA countries, candidate countries
- **Data Types**: Annual energy balances, electricity production, consumption data
- **Regulation**: Harmonized under Regulation (EC) No 1099/2008

### US RTO Data Sources
- **Source**: Regional Transmission Organization hourly operational data
- **Location**: External cloud storage (iCloud Documents)  
- **Time Period**: 2023-2024 hourly resolution
- **Coverage**: Seven major RTOs covering ~2/3 of US electricity demand
- **Data Resolution**: Hourly timestamps with fuel mix breakdowns
- **Metrics**: Load, prices, generation by fuel type (wind, solar, gas, coal, nuclear, etc.)

## Generated Analysis Products

### Visualization Outputs
- **electricity_vs_gas_renewable_with_fit.pdf**: Comprehensive multi-page regression analysis report

### Analysis Reports
- **Multi-country regression analysis**: Individual country gas price impact assessments
- **RTO comparison tables**: US regional electricity market analysis summaries
- **Statistical correlation studies**: European and US market renewable energy impact quantification

## Technical Requirements

### Core Dependencies
```
pandas>=1.3.0          # Data manipulation and analysis
numpy>=1.20.0          # Numerical computing
matplotlib>=3.5.0      # Plotting and visualization  
seaborn>=0.11.0        # Statistical data visualization
scipy>=1.7.0           # Scientific computing and statistics
pathlib                # File system path handling
warnings               # Clean console output
```

### Additional Dependencies
```
scikit-learn>=1.0.0    # Machine learning (LinearRegression, r2_score)
openpyxl>=3.0.0        # Excel file reading (.xlsx support)
```

### Installation
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn openpyxl
```

## Analysis Methodology

### 1. Correlation Analysis
Direct statistical correlation between renewable energy penetration and electricity prices across different markets and time periods.

### 2. Regression Modeling  
**Primary Model**: Linear regression analysis focusing on natural gas prices as the dominant driver of electricity costs in European markets.

**Rationale**: Gas-fired power plants often set marginal pricing in European electricity markets due to merit order dispatch principles.

### 3. Load-Weighted Analysis
**US RTO Analysis**: Uses load-weighted price calculations rather than simple averages to reflect actual market conditions and electricity consumption patterns.

### 4. Cross-Regional Comparison
Comparative analysis methodology:
- **European Analysis**: Country-level aggregated annual data
- **US Analysis**: Regional market-level hourly data with load-weighting
- **Temporal Analysis**: Multi-year comparisons to identify trends

## Research Implications

### Key Findings Framework
The analysis framework enables investigation of:

1. **Merit Order Effects**: How renewable energy (zero marginal cost) affects wholesale electricity pricing
2. **Market Structure**: Differences between European country-level and US regional market dynamics
3. **Primary Cost Drivers**: Quantification of natural gas price influence on electricity markets

### Analytical Robustness
- **Multiple Data Sources**: Cross-validation using Ember, Eurostat, World Bank, and RTO data
- **Statistical Validation**: Correlation analysis, regression diagnostics, R² reporting
- **Temporal Consistency**: Multi-year analysis to account for market variations
- **Geographic Coverage**: European country-level and US regional market analysis

## Data Update and Maintenance

### Update Frequencies
- **European wholesale prices**: Monthly updates available from Ember Energy
- **Renewable share data**: Annual updates from Our World in Data
- **Eurostat data**: Annual updates for retail prices
- **US RTO data**: Hourly updates (analysis typically uses annual aggregations)

### Data Quality Notes
- **Missing Data Handling**: Scripts include comprehensive data validation and missing value handling
- **Country Name Standardization**: Automatic mapping for different naming conventions across datasets
- **Unit Consistency**: Automatic conversion between EUR/MWh and EUR/kWh where appropriate
- **Temporal Alignment**: Careful handling of different data release schedules and time periods

## License and Attribution

This project is released under the [LICENSE](LICENSE) terms. When using this analysis or its outputs:

1. **Data Attribution**: Cite original data sources (Ember, Eurostat, Our World in Data, etc.)
2. **Methodology Reference**: Reference the analytical approaches used
3. **Source Code**: Acknowledge the analysis framework if used for derivative work

### Data Source Citations
- Ember Energy datasets: Follow Ember's attribution requirements
- Our World in Data: Follow OWID's Creative Commons licensing
- Eurostat: Reference official Eurostat databases and regulation compliance
- World Bank: Follow World Bank Open Data terms of use
