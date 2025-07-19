import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_retail_price_vs_renewables(year, retail_excel, renew_csv):
    # Load retail price data
    df_retail = pd.read_excel(retail_excel, sheet_name='Sheet 1', skiprows=10)
    df_retail = df_retail.dropna(subset=[df_retail.columns[0]])
    df_retail = df_retail[df_retail.iloc[:, 0] != 'GEO (Labels)']
    
    # Get retail prices for the specified year
    year_col = str(year)
    if year_col not in df_retail.columns:
        print(f"Year {year} not found in retail data")
        return
    
    retail_data = df_retail[[df_retail.columns[0], year_col]].copy()
    retail_data.columns = ['Country', f'Retail_Price_{year}']
    retail_data = retail_data.dropna(subset=[f'Retail_Price_{year}'])
    
    # Convert price to float (some values might be strings)
    retail_data[f'Retail_Price_{year}'] = pd.to_numeric(retail_data[f'Retail_Price_{year}'], errors='coerce')
    retail_data = retail_data.dropna(subset=[f'Retail_Price_{year}'])
    
    # Load and prepare renewables data
    df_renew = pd.read_csv(renew_csv)
    df_renew_year = df_renew[df_renew['Year'] == year]
    df_renew_year = df_renew_year.rename(columns={'Entity': 'Country', 'Solar and wind - % electricity': f'Renewable_Fraction_{year}'})
    renew_year = df_renew_year[['Country', f'Renewable_Fraction_{year}']]
    
    # Create country mapping for name differences
    country_mapping = {
        'Türkiye': 'Turkey'
    }
    
    # Apply country mapping to retail data
    retail_data['Country'] = retail_data['Country'].replace(country_mapping)
    
    # Merge on country
    df_merge = pd.merge(retail_data, renew_year, on='Country', how='inner')
    
    # Scatter plot with labels
    plt.figure(figsize=(12, 8))
    
    # Extract x and y data for regression, removing NaN values
    df_clean = df_merge.dropna(subset=[f'Renewable_Fraction_{year}', f'Retail_Price_{year}'])
    x_data = df_clean[f'Renewable_Fraction_{year}']
    y_data = df_clean[f'Retail_Price_{year}']
    
    # Calculate least squares fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
    
    # Create fit line
    x_fit = np.linspace(x_data.min(), x_data.max(), 100)
    y_fit = slope * x_fit + intercept
    
    # Plot fit line first (behind scatter points)
    plt.plot(x_fit, y_fit, 'r--', alpha=0.8, linewidth=3, 
             label=f'Fit: y = {slope:.4f}x + {intercept:.3f} (R = {r_value:.3f})')
    # Plot scatter points on top
    plt.scatter(x_data, y_data, alpha=0.7, s=200, edgecolors='black', linewidth=1, color='steelblue')
    
    # Add country labels
    for _, row in df_clean.iterrows():
        plt.annotate(
            row['Country'],
            (row[f'Renewable_Fraction_{year}'], row[f'Retail_Price_{year}']),
            textcoords="offset points",
            xytext=(8, 8),
            ha='left',
            fontsize=11,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9)
        )
    
    # Set y-axis to start at zero
    plt.ylim(bottom=0)
    
    # Format axes
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.0f}%'))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'€{x:.3f}'))
    
    plt.xlabel(f'Renewable Fraction ({year})', fontsize=14)
    plt.ylabel('Retail Price (EUR/kWh)', fontsize=14)
    plt.title(f'Retail Price vs. Renewable Share, {year}', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.show()

def plot_price_difference_vs_renewables(year, retail_excel, price_csv, renew_csv):
    # Load retail price data
    df_retail = pd.read_excel(retail_excel, sheet_name='Sheet 1', skiprows=10)
    df_retail = df_retail.dropna(subset=[df_retail.columns[0]])
    df_retail = df_retail[df_retail.iloc[:, 0] != 'GEO (Labels)']
    
    # Get retail prices for the specified year
    year_col = str(year)
    if year_col not in df_retail.columns:
        print(f"Year {year} not found in retail data")
        return
    
    retail_data = df_retail[[df_retail.columns[0], year_col]].copy()
    retail_data.columns = ['Country', f'Retail_Price_{year}']
    retail_data = retail_data.dropna(subset=[f'Retail_Price_{year}'])
    
    # Convert price to float (some values might be strings)
    retail_data[f'Retail_Price_{year}'] = pd.to_numeric(retail_data[f'Retail_Price_{year}'], errors='coerce')
    retail_data = retail_data.dropna(subset=[f'Retail_Price_{year}'])
    
    # Load wholesale price data
    df_price = pd.read_csv(price_csv)
    df_price['Date'] = pd.to_datetime(df_price['Date'])
    df_year = df_price[df_price['Date'].dt.year == year]
    avg_price = (
        df_year
        .groupby('Country')['Price (EUR/MWhe)']
        .mean()
        .reset_index()
        .rename(columns={'Price (EUR/MWhe)': f'Wholesale_Price_{year}'})
    )
    
    # Convert wholesale price from EUR/MWh to EUR/kWh for comparison
    avg_price[f'Wholesale_Price_{year}'] = avg_price[f'Wholesale_Price_{year}'] / 1000
    
    # Load and prepare renewables data
    df_renew = pd.read_csv(renew_csv)
    df_renew_year = df_renew[df_renew['Year'] == year]
    df_renew_year = df_renew_year.rename(columns={'Entity': 'Country', 'Solar and wind - % electricity': f'Renewable_Fraction_{year}'})
    renew_year = df_renew_year[['Country', f'Renewable_Fraction_{year}']]
    
    # Create country mapping for name differences
    country_mapping = {
        'Türkiye': 'Turkey'
    }
    
    # Apply country mapping to retail data
    retail_data['Country'] = retail_data['Country'].replace(country_mapping)
    
    # Merge all data
    df_merge = pd.merge(retail_data, avg_price, on='Country', how='inner')
    df_merge = pd.merge(df_merge, renew_year, on='Country', how='inner')
    
    # Calculate price difference (retail - wholesale)
    df_merge[f'Price_Difference_{year}'] = df_merge[f'Retail_Price_{year}'] - df_merge[f'Wholesale_Price_{year}']
    
    # Scatter plot with labels
    plt.figure(figsize=(12, 8))
    
    # Extract x and y data for regression, removing NaN values
    df_clean = df_merge.dropna(subset=[f'Renewable_Fraction_{year}', f'Price_Difference_{year}'])
    x_data = df_clean[f'Renewable_Fraction_{year}']
    y_data = df_clean[f'Price_Difference_{year}']
    
    # Calculate least squares fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
    
    # Create fit line
    x_fit = np.linspace(x_data.min(), x_data.max(), 100)
    y_fit = slope * x_fit + intercept
    
    # Plot fit line first (behind scatter points)
    plt.plot(x_fit, y_fit, 'r--', alpha=0.8, linewidth=3, 
             label=f'Fit: y = {slope:.4f}x + {intercept:.3f} (R = {r_value:.3f})')
    # Plot scatter points on top
    plt.scatter(x_data, y_data, alpha=0.7, s=200, edgecolors='black', linewidth=1, color='orange')
    
    # Add country labels
    for _, row in df_clean.iterrows():
        plt.annotate(
            row['Country'],
            (row[f'Renewable_Fraction_{year}'], row[f'Price_Difference_{year}']),
            textcoords="offset points",
            xytext=(8, 8),
            ha='left',
            fontsize=11,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9)
        )
    
    # Set y-axis to start at zero
    plt.ylim(bottom=0)
    
    # Format axes
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.0f}%'))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'€{x:.3f}'))
    
    plt.xlabel(f'Renewable Fraction ({year})', fontsize=14)
    plt.ylabel('Taxes + Distribution (Retail - Wholesale, EUR/kWh)', fontsize=14)
    plt.title(f'Taxes + Distribution vs. Renewable Share, {year}', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.show()

def plot_price_vs_renewables_retail_scale(year, price_csv, renew_csv, retail_excel):
    # First get retail price range for y-axis scaling
    df_retail = pd.read_excel(retail_excel, sheet_name='Sheet 1', skiprows=10)
    df_retail = df_retail.dropna(subset=[df_retail.columns[0]])
    df_retail = df_retail[df_retail.iloc[:, 0] != 'GEO (Labels)']
    
    year_col = str(year)
    if year_col not in df_retail.columns:
        print(f"Year {year} not found in retail data")
        return
    
    retail_data = df_retail[[df_retail.columns[0], year_col]].copy()
    retail_data.columns = ['Country', f'Retail_Price_{year}']
    retail_data = retail_data.dropna(subset=[f'Retail_Price_{year}'])
    retail_data[f'Retail_Price_{year}'] = pd.to_numeric(retail_data[f'Retail_Price_{year}'], errors='coerce')
    retail_data = retail_data.dropna(subset=[f'Retail_Price_{year}'])
    
    # Get retail price range
    retail_min = retail_data[f'Retail_Price_{year}'].min()
    retail_max = retail_data[f'Retail_Price_{year}'].max()
    
    # Load and filter wholesale price data
    df_price = pd.read_csv(price_csv)
    df_price['Date'] = pd.to_datetime(df_price['Date'])
    df_year = df_price[df_price['Date'].dt.year == year]
    avg_price = (
        df_year
        .groupby('Country')['Price (EUR/MWhe)']
        .mean()
        .reset_index()
        .rename(columns={'Price (EUR/MWhe)': f'Avg_Price_{year}'})
    )
    
    # Convert wholesale price from EUR/MWh to EUR/kWh for consistency
    avg_price[f'Avg_Price_{year}'] = avg_price[f'Avg_Price_{year}'] / 1000

    # Load and prepare renewables data
    df_renew = pd.read_csv(renew_csv)
    df_renew_year = df_renew[df_renew['Year'] == year]
    df_renew_year = df_renew_year.rename(columns={'Entity': 'Country', 'Solar and wind - % electricity': f'Renewable_Fraction_{year}'})
    renew_year = df_renew_year[['Country', f'Renewable_Fraction_{year}']]

    # Merge on country
    df_merge = pd.merge(avg_price, renew_year, on='Country', how='inner')

    # Scatter plot with labels
    plt.figure(figsize=(12, 8))
    
    # Extract x and y data for regression, removing NaN values
    df_clean = df_merge.dropna(subset=[f'Renewable_Fraction_{year}', f'Avg_Price_{year}'])
    x_data = df_clean[f'Renewable_Fraction_{year}']
    y_data = df_clean[f'Avg_Price_{year}']
    
    # Calculate least squares fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
    
    # Create fit line
    x_fit = np.linspace(x_data.min(), x_data.max(), 100)
    y_fit = slope * x_fit + intercept
    
    # Plot fit line first (behind scatter points)
    plt.plot(x_fit, y_fit, 'r--', alpha=0.8, linewidth=3, 
             label=f'Fit: y = {slope:.6f}x + {intercept:.3f} (R = {r_value:.3f})')
    # Plot scatter points on top
    plt.scatter(x_data, y_data, alpha=0.7, s=200, edgecolors='black', linewidth=1, color='steelblue')
    
    # Add country labels
    for _, row in df_clean.iterrows():
        plt.annotate(
            row['Country'],
            (row[f'Renewable_Fraction_{year}'], row[f'Avg_Price_{year}']),
            textcoords="offset points",
            xytext=(8, 8),
            ha='left',
            fontsize=11,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9)
        )
    
    # Set y-axis to start at zero, max at retail price range
    plt.ylim(0, retail_max)
    
    # Format axes
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.0f}%'))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'€{x:.3f}'))
    
    plt.xlabel(f'Renewable Fraction ({year})', fontsize=14)
    plt.ylabel('Average Wholesale Price (EUR/kWh)', fontsize=14)
    plt.title(f'Wholesale Price vs. Renewable Share (Retail Scale), {year}', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.show()

def plot_price_vs_renewables(year, price_csv, renew_csv):
    # Load and filter price data
    df_price = pd.read_csv(price_csv)
    df_price['Date'] = pd.to_datetime(df_price['Date'])
    df_year = df_price[df_price['Date'].dt.year == year]
    avg_price = (
        df_year
        .groupby('Country')['Price (EUR/MWhe)']
        .mean()
        .reset_index()
        .rename(columns={'Price (EUR/MWhe)': f'Avg_Price_{year}'})
    )
    
    # Convert wholesale price from EUR/MWh to EUR/kWh for consistency
    avg_price[f'Avg_Price_{year}'] = avg_price[f'Avg_Price_{year}'] / 1000

    # Load and prepare renewables data
    df_renew = pd.read_csv(renew_csv)
    df_renew_year = df_renew[df_renew['Year'] == year]
    df_renew_year = df_renew_year.rename(columns={'Entity': 'Country', 'Solar and wind - % electricity': f'Renewable_Fraction_{year}'})
    renew_year = df_renew_year[['Country', f'Renewable_Fraction_{year}']]

    # Merge on country
    df_merge = pd.merge(avg_price, renew_year, on='Country', how='inner')

    # Scatter plot with labels
    plt.figure(figsize=(12, 8))
    
    # Extract x and y data for regression, removing NaN values
    df_clean = df_merge.dropna(subset=[f'Renewable_Fraction_{year}', f'Avg_Price_{year}'])
    x_data = df_clean[f'Renewable_Fraction_{year}']
    y_data = df_clean[f'Avg_Price_{year}']
    
    # Calculate least squares fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
    
    # Create fit line
    x_fit = np.linspace(x_data.min(), x_data.max(), 100)
    y_fit = slope * x_fit + intercept
    
    # Plot fit line first (behind scatter points)
    plt.plot(x_fit, y_fit, 'r--', alpha=0.8, linewidth=3, 
             label=f'Fit: y = {slope:.6f}x + {intercept:.3f} (R = {r_value:.3f})')
    # Plot scatter points on top
    plt.scatter(x_data, y_data, alpha=0.7, s=200, edgecolors='black', linewidth=1, color='steelblue')
    
    # Add country labels
    for _, row in df_clean.iterrows():
        plt.annotate(
            row['Country'],
            (row[f'Renewable_Fraction_{year}'], row[f'Avg_Price_{year}']),
            textcoords="offset points",
            xytext=(8, 8),
            ha='left',
            fontsize=11,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9)
        )
    
    # Set y-axis to start at zero
    plt.ylim(bottom=0)
    
    # Format axes
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.0f}%'))
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'€{x:.3f}'))
    
    plt.xlabel(f'Renewable Fraction ({year})', fontsize=14)
    plt.ylabel('Average Wholesale Price (EUR/kWh)', fontsize=14)
    plt.title(f'Wholesale Price vs. Renewable Share, {year}', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    plt.show()

# Paths to your data files
price_csv = 'data/european_wholesale_electricity_price_data_monthly.csv'
renew_csv = 'data/share-of-electricity-production-from-solar-and-wind/share-of-electricity-production-from-solar-and-wind.csv'
retail_excel = 'data/ten00117_page_spreadsheet.xlsx'

# Generate the wholesale price plots
# plot_price_vs_renewables(2022, price_csv, renew_csv)
plot_price_vs_renewables(2023, price_csv, renew_csv)

# Generate the retail price plots
# plot_retail_price_vs_renewables(2022, retail_excel, renew_csv)
plot_retail_price_vs_renewables(2023, retail_excel, renew_csv)

# Generate the price difference plots
# plot_price_difference_vs_renewables(2022, retail_excel, price_csv, renew_csv)
plot_price_difference_vs_renewables(2023, retail_excel, price_csv, renew_csv)

# Generate the wholesale price plot with retail scale
# plot_price_vs_renewables_retail_scale(2022, price_csv, renew_csv, retail_excel)
plot_price_vs_renewables_retail_scale(2023, price_csv, renew_csv, retail_excel)
