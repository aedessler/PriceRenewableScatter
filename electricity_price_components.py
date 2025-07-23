#!/usr/bin/env python3
"""
Electricity Price Components Analysis

This script creates a stacked bar chart showing electricity price components 
(generation, network, and taxes) for European countries using Eurostat data
from the nrg_pc_204_c dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("Set2")

def load_electricity_price_components(file_path, year=2023, consumption_band='DB'):
    """
    Load electricity price components from the Eurostat Excel file.
    
    Parameters:
    -----------
    file_path : str
        Path to the nrg_pc_204_c Excel file
    year : int
        Year to extract data for (default: 2023)
    consumption_band : str
        'DA' for <1000 kWh or 'DB' for 1000-2499 kWh (default: 'DB')
    
    Returns:
    --------
    dict
        Dictionary with DataFrames for each price component
    """
    
    # Define the sheet mapping based on consumption band
    if consumption_band == 'DA':  # <1000 kWh
        sheet_mapping = {
            'energy_supply': 'Sheet 1',
            'network_costs': 'Sheet 2', 
            'taxes_fees_levies': 'Sheet 3',
            'vat': 'Sheet 4',
            'renewable_taxes': 'Sheet 5',
            'capacity_charges': 'Sheet 6',
            'environmental_taxes': 'Sheet 7',
            'nuclear_taxes': 'Sheet 8',
            'allowances_balancing': 'Sheet 9',
            'other_taxes_levies': 'Sheet 15'
        }
    else:  # DB: 1000-2499 kWh
        sheet_mapping = {
            'energy_supply': 'Sheet 16',
            'network_costs': 'Sheet 17',
            'taxes_fees_levies': 'Sheet 18', 
            'vat': 'Sheet 19',
            'renewable_taxes': 'Sheet 20',
            'capacity_charges': 'Sheet 21',
            'environmental_taxes': 'Sheet 22',
            'nuclear_taxes': 'Sheet 23',
            'allowances_balancing': 'Sheet 24',
            'other_taxes_levies': 'Sheet 30'
        }
    
    components = {}
    
    try:
        for component, sheet_name in sheet_mapping.items():
            # Read the data sheet with proper header row (row 9 contains the headers)
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=9)
            
            # Skip the first row after header which contains "GEO (Labels)"
            df = df.iloc[1:].reset_index(drop=True)
            
            # Clean up the dataframe
            # The first column contains country names
            df = df.rename(columns={df.columns[0]: 'Country'})
            
            # Remove rows with invalid country data
            df = df[df['Country'].notna()]
            df = df[~df['Country'].isin(['Special value', ':', 'nan'])]
            df = df[df['Country'] != '']
            
            # Convert year columns to proper format and find the target year
            year_str = str(year)
            year_float = float(year)
            
            # Check both string and float versions of the year
            target_column = None
            if year_str in df.columns:
                target_column = year_str
            elif year_float in df.columns:
                target_column = year_float
            
            if target_column is not None:
                # Convert the year data to numeric
                df[target_column] = pd.to_numeric(df[target_column], errors='coerce')
                # Keep only Country and the target year
                df = df[['Country', target_column]].copy()
                df = df.rename(columns={target_column: component})
                # Remove rows with missing data for this component
                df = df.dropna(subset=[component])
                components[component] = df
            else:
                available_years = [col for col in df.columns if str(col).isdigit() or (isinstance(col, float) and not pd.isna(col))]
                print(f"Warning: Year {year} not found in {component} data. Available years: {available_years}")
                    
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return {}
    
    return components

def load_total_costs_from_ten00117(file_path, year=2024, consumption_band='DB'):
    """
    Load total electricity costs from the ten00117 Excel file.
    
    Parameters:
    -----------
    file_path : str
        Path to the ten00117 Excel file
    year : int
        Year to extract data for (default: 2024)
    consumption_band : str
        'DA' for <1000 kWh or 'DB' for 1000-2499 kWh (default: 'DB')
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with countries and total costs
    """
    
    try:
        # Load retail price data from ten00117 file
        df_retail = pd.read_excel(file_path, sheet_name='Sheet 1', skiprows=10)
        df_retail = df_retail.dropna(subset=[df_retail.columns[0]])
        df_retail = df_retail[df_retail.iloc[:, 0] != 'GEO (Labels)']
        
        # Get retail prices for the specified year
        year_col = str(year)
        if year_col not in df_retail.columns:
            print(f"Warning: Year {year} not found in ten00117 data. Available years: {[col for col in df_retail.columns if str(col).isdigit()]}")
            return pd.DataFrame()
        
        retail_data = df_retail[[df_retail.columns[0], year_col]].copy()
        retail_data.columns = ['Country', 'Total_Cost']
        retail_data = retail_data.dropna(subset=['Total_Cost'])
        
        # Convert price to float (some values might be strings)
        retail_data['Total_Cost'] = pd.to_numeric(retail_data['Total_Cost'], errors='coerce')
        retail_data = retail_data.dropna(subset=['Total_Cost'])
        
        # Apply country mapping for consistency
        country_label_mapping = {
            'Euro area (EA11-1999, EA12-2001, EA13-2007, EA15-2008, EA16-2009, EA17-2011, EA18-2014, EA19-2015, EA20-2023)': 'Euro Area',
            'European Union - 27 countries (from 2020)': 'EU-27',
            'Türkiye': 'Turkey'
        }
        retail_data['Country'] = retail_data['Country'].replace(country_label_mapping)
        
        return retail_data
        
    except Exception as e:
        print(f"Error reading ten00117 file: {e}")
        return pd.DataFrame()

def combine_components_data(components, year=2023, consumption_band='DB'):
    """
    Combine all price components into a single DataFrame using total costs from ten00117.
    
    Parameters:
    -----------
    components : dict
        Dictionary of DataFrames with price components
    year : int
        Year for the analysis
    consumption_band : str
        Consumption band for the analysis
        
    Returns:
    --------
    pd.DataFrame
        Combined DataFrame with all components
    """
    
    if not components:
        return pd.DataFrame()
    
    # Load total costs from ten00117 file
    ten00117_file = 'data/ten00117_page_spreadsheet.xlsx'
    total_costs_df = load_total_costs_from_ten00117(ten00117_file, year, consumption_band)
    
    if total_costs_df.empty:
        print("Warning: Could not load total costs from ten00117 file")
        return pd.DataFrame()
    
    # Start with generation and network components only
    generation_df = components.get('energy_supply')
    network_df = components.get('network_costs')
    
    if generation_df is None or network_df is None:
        print("Warning: Missing generation or network cost data")
        return pd.DataFrame()
    
    # Merge generation and network costs
    merged_df = pd.merge(generation_df, network_df, on='Country', how='outer')
    merged_df = merged_df.rename(columns={
        'energy_supply': 'Generation',
        'network_costs': 'Network'
    })
    
    # Merge with total costs from ten00117
    merged_df = pd.merge(merged_df, total_costs_df, on='Country', how='inner')
    merged_df = merged_df.rename(columns={'Total_Cost': 'Total'})
    
    # Fill NaN values with 0 for generation and network
    merged_df['Generation'] = merged_df['Generation'].fillna(0)
    merged_df['Network'] = merged_df['Network'].fillna(0)
    
    # Calculate taxes as the difference: Total - Generation - Network
    merged_df['Taxes'] = merged_df['Total'] - merged_df['Generation'] - merged_df['Network']
    
    # Keep only relevant columns
    result_df = merged_df[['Country', 'Generation', 'Network', 'Taxes', 'Total']].copy()
    
    # Shorten long country labels for better chart readability
    country_label_mapping = {
        'Euro area (EA11-1999, EA12-2001, EA13-2007, EA15-2008, EA16-2009, EA17-2011, EA18-2014, EA19-2015, EA20-2023)': 'Euro Area',
        'European Union - 27 countries (from 2020)': 'EU-27'
    }
    
    result_df['Country'] = result_df['Country'].replace(country_label_mapping)
    
    # Remove rows where all components are zero
    result_df = result_df[(result_df['Generation'] != 0) | 
                         (result_df['Network'] != 0) | 
                         (result_df['Taxes'] != 0)]
    
    return result_df

def create_stacked_bar_chart(df, year=2023, consumption_band='DB', save_plot=True):
    """
    Create a diverging bar chart of electricity price components by country.
    Positive components (generation, network) stack to the right.
    Negative taxes (subsidies) extend to the left.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with price components by country
    year : int
        Year for the chart title
    consumption_band : str
        Consumption band for the chart title
    save_plot : bool
        Whether to save the plot as PNG
    """
    
    if df.empty:
        print("No data available for plotting.")
        return
    
    # Sort countries by total price (ascending)
    df_sorted = df.sort_values('Total', ascending=True)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Define colors for each component
    colors = {
        'Generation': '#2E8B57',    # Sea Green
        'Network': '#4682B4',       # Steel Blue  
        'Taxes_Positive': '#CD5C5C',  # Indian Red (for positive taxes)
        'Taxes_Negative': '#CD5C5C'   # Same as positive taxes (Indian Red)
    }
    
    # Extract data
    countries = df_sorted['Country']
    generation = df_sorted['Generation']
    network = df_sorted['Network'] 
    taxes = df_sorted['Taxes']
    
    # Separate positive and negative taxes
    taxes_positive = taxes.clip(lower=0)  # Only positive values
    taxes_negative = taxes.clip(upper=0)  # Only negative values (subsidies)
    
    # Create the bars
    # 1. Generation costs (starting from 0)
    bars1 = ax.barh(countries, generation, label='Generation & Supply', 
                    color=colors['Generation'], alpha=0.8)
    
    # 2. Network costs (stacked on generation)
    bars2 = ax.barh(countries, network, left=generation, label='Network Costs',
                    color=colors['Network'], alpha=0.8)
    
    # 3. Positive taxes (stacked on generation + network)
    positive_base = generation + network
    bars3 = ax.barh(countries, taxes_positive, left=positive_base, 
                    label='Taxes & Levies', color=colors['Taxes_Positive'], alpha=0.8)
    
    # 4. Negative taxes/subsidies (extending to the left from 0) - no label for legend
    bars4 = ax.barh(countries, taxes_negative, 
                    color=colors['Taxes_Negative'], alpha=0.8)
    
    # Customize the plot
    consumption_desc = "< 1,000 kWh" if consumption_band == 'DA' else "1,000-2,499 kWh"
    ax.set_xlabel('Price Components (EUR/kWh)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Country', fontsize=12, fontweight='bold')
    ax.set_title(f'Electricity Price Components by Country ({year})\n'
                f'Household Consumption: {consumption_desc} annually\n'
                f'(Subsidies shown as negative values to the left)', 
                fontsize=14, fontweight='bold', pad=25)
    
    # Add a vertical line at x=0 to emphasize the baseline
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.7)
    
    # Add legend
    ax.legend(loc='lower right', fontsize=11)
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='x')
    
    # Format x-axis to show currency (with proper handling of negative values)
    def currency_formatter(x, p):
        if x >= 0:
            return f'€{x:.3f}'
        else:
            return f'-€{abs(x):.3f}'
    
    ax.xaxis.set_major_formatter(plt.FuncFormatter(currency_formatter))
    
    # Adjust layout
    plt.tight_layout()
    
    # Add value labels on bars for better readability
    for i, (country, gen, net, tax_pos, tax_neg) in enumerate(zip(countries, generation, network, taxes_positive, taxes_negative)):
        total = gen + net + tax_pos + tax_neg  # tax_neg is already negative
        
        # For countries with subsidies (negative taxes), show total at right edge of positive components
        if tax_neg < -0.001:  # Has significant subsidies
            # Show total cost at the right edge of the positive components (gen + net + positive taxes)
            right_edge = gen + net + tax_pos
            ax.text(right_edge + 0.005, i, f'€{total:.3f}', 
                    va='center', ha='left', fontsize=9, fontweight='bold')
        else:
            # For countries without subsidies, show total at the end of all components
            ax.text(total + 0.005, i, f'€{total:.3f}', 
                    va='center', ha='left', fontsize=9, fontweight='bold')
        
        # Remove subsidy labels as requested
    
    # Extend x-axis limits to accommodate negative values and labels
    x_min = min(0, df_sorted['Taxes'].min() - 0.02)
    x_max = df_sorted['Total'].max() + 0.05
    ax.set_xlim(x_min, x_max)
    
    # Save the plot
    if save_plot:
        filename = f'electricity_price_components_{year}_{consumption_band}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Chart saved as {filename}")
    
    plt.close()

def print_summary_statistics(df, year=2023, consumption_band='DB'):
    """
    Print summary statistics for the price components.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with price components
    year : int
        Year for the summary
    consumption_band : str
        Consumption band
    """
    
    if df.empty:
        print("No data available for summary.")
        return
    
    consumption_desc = "< 1,000 kWh" if consumption_band == 'DA' else "1,000-2,499 kWh"
    
    print(f"\n{'='*60}")
    print(f"ELECTRICITY PRICE COMPONENTS SUMMARY ({year})")
    print(f"Household Consumption: {consumption_desc} annually")
    print(f"{'='*60}")
    
    print(f"\nNumber of countries analyzed: {len(df)}")
    
    # Calculate statistics
    stats = df[['Generation', 'Network', 'Taxes', 'Total']].describe()
    
    print(f"\nPRICE COMPONENT STATISTICS (EUR/kWh):")
    print(f"{'Component':<15} {'Mean':<8} {'Median':<8} {'Min':<8} {'Max':<8}")
    print("-" * 55)
    
    for component in ['Generation', 'Network', 'Taxes', 'Total']:
        mean_val = df[component].mean()
        median_val = df[component].median()
        min_val = df[component].min()
        max_val = df[component].max()
        
        print(f"{component:<15} {mean_val:<8.3f} {median_val:<8.3f} {min_val:<8.3f} {max_val:<8.3f}")
    
    # Component shares
    print(f"\nAVERAGE COMPONENT SHARES:")
    total_avg = df['Total'].mean()
    gen_share = (df['Generation'].mean() / total_avg) * 100
    net_share = (df['Network'].mean() / total_avg) * 100
    tax_share = (df['Taxes'].mean() / total_avg) * 100
    
    print(f"Generation & Supply: {gen_share:.1f}%")
    print(f"Network Costs:      {net_share:.1f}%") 
    print(f"Taxes & Levies:     {tax_share:.1f}%")
    
    # Top and bottom countries by total price
    print(f"\nTOP 5 MOST EXPENSIVE:")
    top_5 = df.nlargest(5, 'Total')[['Country', 'Total']]
    for _, row in top_5.iterrows():
        print(f"  {row['Country']}: €{row['Total']:.3f}/kWh")
    
    print(f"\nTOP 5 LEAST EXPENSIVE:")
    bottom_5 = df.nsmallest(5, 'Total')[['Country', 'Total']]
    for _, row in bottom_5.iterrows():
        print(f"  {row['Country']}: €{row['Total']:.3f}/kWh")

def load_renewable_data(file_path, year=2024):
    """
    Load renewable energy share data from CSV file.
    
    Parameters:
    -----------
    file_path : str
        Path to the renewable energy share CSV file
    year : int
        Year to extract data for
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with countries and renewable share
    """
    
    try:
        df = pd.read_csv(file_path)
        
        # Filter for the specified year
        df_year = df[df['Year'] == year].copy()
        
        if df_year.empty:
            available_years = sorted(df['Year'].unique())
            print(f"Warning: Year {year} not found in renewable data. Available years: {available_years}")
            return pd.DataFrame()
        
        # Clean up country names and select relevant columns
        df_year = df_year[['Entity', 'Solar and wind - % electricity']].copy()
        df_year = df_year.rename(columns={
            'Entity': 'Country', 
            'Solar and wind - % electricity': 'Renewable_Share'
        })
        
        # Remove rows with missing data
        df_year = df_year.dropna(subset=['Renewable_Share'])
        
        # Apply country mapping for consistency with price data
        country_mapping = {
            'United Kingdom': 'United Kingdom',
            'Czech Republic': 'Czechia',
            'Slovak Republic': 'Slovakia'
        }
        df_year['Country'] = df_year['Country'].replace(country_mapping)
        
        return df_year
        
    except Exception as e:
        print(f"Error reading renewable data file: {e}")
        return pd.DataFrame()

def create_renewable_vs_network_plot(df, renewable_df, year=2024, save_plot=True):
    """
    Create a scatter plot of renewable fraction vs network costs.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with electricity price components
    renewable_df : pd.DataFrame
        DataFrame with renewable energy shares
    year : int
        Year for the chart title
    save_plot : bool
        Whether to save the plot as PNG
    """
    
    if df.empty or renewable_df.empty:
        print("No data available for renewable vs network plot.")
        return
    
    # Merge the datasets
    merged_df = pd.merge(df[['Country', 'Network']], renewable_df, on='Country', how='inner')
    
    if merged_df.empty:
        print("No matching countries found between price and renewable data.")
        return
    
    # Create the scatter plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot
    scatter = ax.scatter(merged_df['Renewable_Share'], merged_df['Network'], 
                        alpha=0.7, s=80, color='#4682B4', edgecolors='black', linewidth=0.5)
    
    # Add country labels
    for _, row in merged_df.iterrows():
        ax.annotate(row['Country'], 
                   (row['Renewable_Share'], row['Network']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, alpha=0.8)
    
    # Add least squares fit
    x = merged_df['Renewable_Share'].values
    y = merged_df['Network'].values
    
    # Calculate linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Create line for plotting
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept
    
    # Plot the regression line
    ax.plot(x_line, y_line, color='red', linestyle='--', linewidth=2, alpha=0.8, 
            label=f'Linear fit: R² = {r_value**2:.3f}, p = {p_value:.3f}')
    
    # Add legend for the fit line
    ax.legend(loc='best', fontsize=10)
    
    # Customize the plot
    ax.set_xlabel('Renewable Energy Share (Solar + Wind, %)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Network Costs (EUR/kWh)', fontsize=12, fontweight='bold')
    ax.set_title(f'Renewable Energy Share vs Network Costs ({year})\n'
                f'European Countries', fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Format y-axis as currency
    def currency_formatter(x, p):
        return f'€{x:.3f}'
    ax.yaxis.set_major_formatter(plt.FuncFormatter(currency_formatter))
    
    # Format x-axis as percentage
    def percentage_formatter(x, p):
        return f'{x:.0f}%'
    ax.xaxis.set_major_formatter(plt.FuncFormatter(percentage_formatter))
    
    # Set y-axis limits
    ax.set_ylim(0, 0.42)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    if save_plot:
        filename = f'renewable_vs_network_costs_{year}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Renewable vs network costs chart saved as {filename}")
    
    plt.close()

def print_data_table(df, year=2023, consumption_band='DB'):
    """
    Print a table of all data used in the chart, sorted alphabetically by country.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with price components
    year : int
        Year for the table title
    consumption_band : str
        Consumption band
    """
    
    if df.empty:
        print("No data available for table.")
        return
    
    consumption_desc = "< 1,000 kWh" if consumption_band == 'DA' else "1,000-2,499 kWh"
    
    print(f"\n{'='*80}")
    print(f"ELECTRICITY PRICE COMPONENTS DATA TABLE ({year})")
    print(f"Household Consumption: {consumption_desc} annually")
    print(f"{'='*80}")
    
    # Sort by country name alphabetically
    df_sorted = df.sort_values('Country').copy()
    
    # Print table header
    print(f"{'Country':<35} {'Generation':<12} {'Network':<12} {'Taxes':<12} {'Total':<12}")
    print(f"{'':35} {'(EUR/kWh)':<12} {'(EUR/kWh)':<12} {'(EUR/kWh)':<12} {'(EUR/kWh)':<12}")
    print("-" * 95)
    
    # Print data rows
    for _, row in df_sorted.iterrows():
        country = row['Country']
        generation = row['Generation']
        network = row['Network']
        taxes = row['Taxes']
        total = row['Total']
        
        # Truncate country name if too long
        if len(country) > 34:
            country = country[:31] + "..."
        
        print(f"{country:<35} {generation:<12.3f} {network:<12.3f} {taxes:<12.3f} {total:<12.3f}")
    
    print("-" * 95)
    print(f"{'TOTAL COUNTRIES:':<35} {len(df_sorted)}")

def main():
    """
    Main function to run the electricity price components analysis.
    """
    
    print("Electricity Price Components Analysis")
    print("="*50)
    
    # File path
    file_path = 'data/nrg_pc_204_c__custom_17509578_spreadsheet.xlsx'
    
    # Check if file exists
    if not Path(file_path).exists():
        print(f"Error: File {file_path} not found!")
        print("Please ensure the data file is in the correct location.")
        return
    
    # Parameters
    year = 2023
    consumption_band = 'DB'  # DB = 1000-2499 kWh consumption band
    
    print(f"Loading electricity price components for {year}...")
    print(f"Consumption band: {'< 1,000 kWh' if consumption_band == 'DA' else '1,000-2,499 kWh'}")
    
    # Load the data
    components = load_electricity_price_components(file_path, year, consumption_band)
    
    if not components:
        print("Failed to load data. Please check the file format and try again.")
        return
    
    print(f"Loaded {len(components)} price components")
    
    # Combine components
    df = combine_components_data(components, year, consumption_band)
    
    if df.empty:
        print("No valid data found for the specified year and consumption band.")
        return
    
    print(f"Successfully processed data for {len(df)} countries")
    
    # Print summary statistics
    print_summary_statistics(df, year, consumption_band)
    
    # Print data table
    print_data_table(df, year, consumption_band)
    
    # Create the visualization
    print(f"\nCreating stacked bar chart...")
    create_stacked_bar_chart(df, year, consumption_band)
    
    # Load renewable data and create renewable vs network plot
    renewable_file = 'data/share-of-electricity-production-from-solar-and-wind/share-of-electricity-production-from-solar-and-wind.csv'
    if Path(renewable_file).exists():
        print(f"\nLoading renewable energy data for {year}...")
        renewable_df = load_renewable_data(renewable_file, year)
        
        if not renewable_df.empty:
            print(f"Creating renewable vs network costs plot...")
            create_renewable_vs_network_plot(df, renewable_df, year)
        else:
            print("No renewable data available for the specified year.")
    else:
        print(f"Warning: Renewable data file {renewable_file} not found.")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()