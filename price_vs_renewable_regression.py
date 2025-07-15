import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Configuration - Set regression type
INCLUDE_RENEWABLE = False  # Set to False for gas-only regression, True for gas + renewable regression
INCLUDE_2022 = False  # Set to False to exclude 2022 data from linear fit (but still show on plot)

def get_natural_gas_data():
    # Read European natural gas prices from CMO Excel file
    header_df = pd.read_excel('CMO-Historical-Data-Monthly.xlsx', sheet_name='Monthly Prices', skiprows=3, nrows=1)
    df = pd.read_excel('CMO-Historical-Data-Monthly.xlsx', sheet_name='Monthly Prices', skiprows=4)
    
    # Set proper column names
    df.columns = header_df.iloc[0].tolist()
    
    # Filter out rows with units (first row after header)
    df = df[df.iloc[:, 0] != df.iloc[:, 0].iloc[0]]
    
    # Extract date and European natural gas price
    gas_data = []
    for _, row in df.iterrows():
        date_str = row.iloc[0]  # First column contains date
        gas_price = row['Natural gas, Europe']
        
        if pd.notna(date_str) and pd.notna(gas_price):
            # Convert date string like '1960M01' to proper date
            year = int(date_str[:4])
            month = int(date_str[5:7])
            gas_data.append({
                'Date': pd.Timestamp(year=year, month=month, day=1),
                'Gas_Price': float(gas_price)
            })
    
    gas_df = pd.DataFrame(gas_data)
    gas_df['Year'] = gas_df['Date'].dt.year
    gas_df['Month'] = gas_df['Date'].dt.month
    
    return gas_df

def get_renewable_data():
    # Read renewable energy data from CSV
    renewable_df = pd.read_csv('share-of-electricity-production-from-solar-and-wind/share-of-electricity-production-from-solar-and-wind.csv')
    
    # Rename columns for consistency
    renewable_df = renewable_df.rename(columns={
        'Entity': 'Country',
        'Solar and wind - % electricity': 'Renewable_Fraction'
    })
    
    # Filter out rows without proper country codes and convert renewable fraction to decimal
    renewable_df = renewable_df[renewable_df['Country'].notna()]
    renewable_df['Renewable_Fraction'] = renewable_df['Renewable_Fraction'] / 100  # Convert percentage to fraction
    
    return renewable_df[['Country', 'Year', 'Renewable_Fraction']]

def load_and_process_data():
    price_df = pd.read_csv('european_wholesale_electricity_price_data_monthly.csv')
    gas_df = get_natural_gas_data()
    renewable_df = get_renewable_data()
    
    price_df['Date'] = pd.to_datetime(price_df['Date'])
    price_df['Year'] = price_df['Date'].dt.year
    
    # Calculate annual averages
    annual_price = price_df.groupby(['Country', 'Year'])['Price (EUR/MWhe)'].mean().reset_index()
    annual_gas = gas_df.groupby('Year')['Gas_Price'].mean().reset_index()
    annual_renewable = renewable_df.groupby(['Country', 'Year'])['Renewable_Fraction'].mean().reset_index()
    
    return annual_price, annual_gas, annual_renewable

def perform_regression_analysis():
    electricity_df, gas_df, renewable_df = load_and_process_data()
    
    # Get all unique countries and sort them, excluding Montenegro and North Macedonia
    all_countries = sorted([country for country in electricity_df['Country'].unique() 
                           if country not in ['Montenegro', 'North Macedonia']])
    
    regression_results = []
    
    print("Least Squares Regression Results:")
    print("=" * 80)
    if INCLUDE_RENEWABLE:
        print("Model: Price = a * Gas_Cost + b * Renewable_Fraction + c")
    else:
        print("Model: Price = a * Gas_Cost + c")
    print("=" * 80)
    
    for country in all_countries:
        # Get data for this country
        country_electricity = electricity_df[electricity_df['Country'] == country]
        
        # Merge electricity and gas data
        merged_data = pd.merge(country_electricity, gas_df, on='Year', how='inner')
        
        # Add renewable data if needed
        if INCLUDE_RENEWABLE:
            country_renewable = renewable_df[renewable_df['Country'] == country]
            merged_data = pd.merge(merged_data, country_renewable, on=['Country', 'Year'], how='inner')
        
        if len(merged_data) >= 3:  # Need at least 3 data points for regression
            # Filter data for regression based on INCLUDE_2022 setting
            if INCLUDE_2022:
                regression_data = merged_data
            else:
                regression_data = merged_data[merged_data['Year'] != 2022]
            
            # Check if we still have enough data points after filtering
            if len(regression_data) >= 3:
                # Prepare data for regression
                if INCLUDE_RENEWABLE:
                    X = regression_data[['Gas_Price', 'Renewable_Fraction']].values
                else:
                    X = regression_data[['Gas_Price']].values
                y = regression_data['Price (EUR/MWhe)'].values
                
                # Perform linear regression
                model = LinearRegression()
                model.fit(X, y)
                
                # Make predictions
                y_pred = model.predict(X)
                
                # Calculate R-squared
                r2 = r2_score(y, y_pred)
                
                # Store results
                result = {
                    'Country': country,
                    'Gas_Coeff': model.coef_[0],
                    'Renewable_Coeff': model.coef_[1] if INCLUDE_RENEWABLE else None,
                    'Intercept': model.intercept_,
                    'R_squared': r2,
                    'Data_Points': len(merged_data),  # Keep original merged_data count for display
                    'Regression_Points': len(regression_data),  # Points used for regression
                    'Years': f"{merged_data['Year'].min()}-{merged_data['Year'].max()}",
                    'Model': model,
                    'Data': merged_data,  # Keep full data for plotting
                    'Include_Renewable': INCLUDE_RENEWABLE
                }
                regression_results.append(result)
                
                # Print results
                print(f"\n{country}:")
                if INCLUDE_RENEWABLE:
                    print(f"  Price = {model.coef_[0]:.3f} * Gas_Cost + {model.coef_[1]:.1f} * Renewable_Fraction + {model.intercept_:.2f}")
                else:
                    print(f"  Price = {model.coef_[0]:.3f} * Gas_Cost + {model.intercept_:.2f}")
                print(f"  R² = {r2:.3f}")
                print(f"  Data points: {len(merged_data)} ({merged_data['Year'].min()}-{merged_data['Year'].max()})")
                if not INCLUDE_2022:
                    print(f"  Regression points: {len(regression_data)} (2022 excluded from fit)")
                
                # Interpret coefficients
                gas_impact = "increases" if model.coef_[0] > 0 else "decreases"
                print(f"  Gas price impact: 1 $/MMBtu {gas_impact} electricity price by {abs(model.coef_[0]):.3f} EUR/MWh")
                if INCLUDE_RENEWABLE:
                    renewable_impact = "increases" if model.coef_[1] > 0 else "decreases"
                    print(f"  Renewable impact: 1% increase in renewables {renewable_impact} electricity price by {abs(model.coef_[1]*0.01):.3f} EUR/MWh")
            else:
                print(f"\n{country}: Insufficient data for regression after filtering (only {len(regression_data)} data points)")
        else:
            print(f"\n{country}: Insufficient data for regression (only {len(merged_data)} data points)")
    
    return regression_results

def plot_regression_results(regression_results):
    # Create multiple plots and save to PDF
    countries_per_plot = 6
    num_plots = (len(regression_results) + countries_per_plot - 1) // countries_per_plot
    
    with PdfPages('electricity_vs_gas_renewable_with_fit.pdf') as pdf:
        for plot_num in range(num_plots):
            start_idx = plot_num * countries_per_plot
            end_idx = min(start_idx + countries_per_plot, len(regression_results))
            results_subset = regression_results[start_idx:end_idx]
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for i, result in enumerate(results_subset):
                country = result['Country']
                data = result['Data']
                model = result['Model']
                
                # Add regression line (behind the data points)
                gas_range = np.linspace(data['Gas_Price'].min(), data['Gas_Price'].max(), 100)
                
                if result['Include_Renewable']:
                    median_renewable = data['Renewable_Fraction'].median()
                    X_pred = np.column_stack([gas_range, np.full(len(gas_range), median_renewable)])
                else:
                    X_pred = gas_range.reshape(-1, 1)
                
                y_pred = model.predict(X_pred)
                
                axes[i].plot(gas_range, y_pred, 'r-', linewidth=2, alpha=0.8, zorder=1,
                           label=f'Fit (R²={result["R_squared"]:.3f})')
                
                # Create scatter plot (on top of the line)
                # Distinguish 2022 data points if they're excluded from regression
                if not INCLUDE_2022 and 2022 in data['Year'].values:
                    data_2022 = data[data['Year'] == 2022]
                    data_other = data[data['Year'] != 2022]
                    
                    # Plot non-2022 data points (used in regression)
                    scatter1 = axes[i].scatter(data_other['Gas_Price'], data_other['Price (EUR/MWhe)'], 
                                             alpha=0.8, s=60, edgecolors='black', color='steelblue', zorder=2,
                                             label='Used in fit')
                    
                    # Plot 2022 data points (excluded from regression) with different color
                    scatter2 = axes[i].scatter(data_2022['Gas_Price'], data_2022['Price (EUR/MWhe)'], 
                                             alpha=0.8, s=60, edgecolors='black', color='red', zorder=2,
                                             label='2022 (excluded from fit)')
                else:
                    # Plot all data points normally
                    scatter = axes[i].scatter(data['Gas_Price'], data['Price (EUR/MWhe)'], 
                                            alpha=0.8, s=60, edgecolors='black', color='steelblue', zorder=2)
                
                # Customize plot
                axes[i].set_xlabel('Annual Avg European Natural Gas Price ($/MMBtu)')
                axes[i].set_ylabel('Annual Avg Electricity Price (EUR/MWh)')
                axes[i].set_title(f'{country}')
                axes[i].grid(True, alpha=0.3)
                axes[i].legend()
                
                # Add slope text on the panel
                gas_slope = result['Gas_Coeff']
                axes[i].text(0.05, 0.95, f'Slope: {gas_slope:.3f}', 
                           transform=axes[i].transAxes, fontsize=10, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                           verticalalignment='top')
                
                # Add year labels
                for _, row in data.iterrows():
                    axes[i].annotate(f'{int(row["Year"])}', 
                                   (row['Gas_Price'], row['Price (EUR/MWhe)']),
                                   xytext=(3, 3), textcoords='offset points', fontsize=7)
                
            
            # Hide unused subplots
            for i in range(len(results_subset), 6):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            if INCLUDE_RENEWABLE:
                plt.suptitle(f'Electricity Price vs Gas Price & Renewable Fraction (Page {plot_num + 1}/{num_plots})', 
                            y=1.02, fontsize=16)
            else:
                plt.suptitle(f'Electricity Price vs Gas Price (Page {plot_num + 1}/{num_plots})', 
                            y=1.02, fontsize=16)
            
            # Save the figure to PDF
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            print(f'Added page {plot_num + 1} to PDF with countries: {", ".join([r["Country"] for r in results_subset])}')
    
    print('Saved all plots to electricity_vs_gas_renewable_with_fit.pdf')

def create_summary_plot(regression_results):
    # Create a summary plot showing gas vs renewable coefficients
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    countries = [r['Country'] for r in regression_results]
    gas_coeffs = [r['Gas_Coeff'] for r in regression_results]
    r2_values = [r['R_squared'] for r in regression_results]
    
    if INCLUDE_RENEWABLE:
        renewable_coeffs = [r['Renewable_Coeff'] for r in regression_results]
        
        # Plot gas coefficients vs renewable coefficients
        scatter1 = ax1.scatter(gas_coeffs, renewable_coeffs, c=r2_values, cmap='viridis', 
                              s=100, alpha=0.7, edgecolors='black')
        ax1.set_xlabel('Gas Price Coefficient (EUR/MWh per $/MMBtu)')
        ax1.set_ylabel('Renewable Fraction Coefficient (EUR/MWh per unit fraction)')
        ax1.set_title('Regression Coefficients by Country')
        ax1.grid(True, alpha=0.3)
        
        # Add country labels
        for i, country in enumerate(countries):
            ax1.annotate(country, (gas_coeffs[i], renewable_coeffs[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.colorbar(scatter1, ax=ax1, label='R²')
    else:
        # Plot gas coefficients vs R² for gas-only regression
        scatter1 = ax1.scatter(gas_coeffs, r2_values, s=100, alpha=0.7, 
                              edgecolors='black', color='steelblue')
        ax1.set_xlabel('Gas Price Coefficient (EUR/MWh per $/MMBtu)')
        ax1.set_ylabel('R² Value')
        ax1.set_title('Gas Price Coefficient vs Model Fit by Country')
        ax1.grid(True, alpha=0.3)
        
        # Add country labels
        for i, country in enumerate(countries):
            ax1.annotate(country, (gas_coeffs[i], r2_values[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot R² values
    ax2.barh(countries, r2_values, color='steelblue', alpha=0.7)
    ax2.set_xlabel('R² Value')
    ax2.set_title('Model Fit Quality by Country')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    # plt.savefig('overall_least_squares_fit.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print('Saved summary plot to overall_least_squares_fit.png')

if __name__ == "__main__":
    # Perform regression analysis
    regression_results = perform_regression_analysis()
    
    if regression_results:
        # Create plots
        plot_regression_results(regression_results)
        create_summary_plot(regression_results)
        
        # Print summary statistics
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS:")
        print("=" * 80)
        avg_gas_coeff = np.mean([r['Gas_Coeff'] for r in regression_results])
        avg_r2 = np.mean([r['R_squared'] for r in regression_results])
        
        print(f"Average gas price coefficient: {avg_gas_coeff:.3f} EUR/MWh per $/MMBtu")
        if INCLUDE_RENEWABLE:
            avg_renewable_coeff = np.mean([r['Renewable_Coeff'] for r in regression_results])
            print(f"Average renewable coefficient: {avg_renewable_coeff:.1f} EUR/MWh per unit fraction")
        print(f"Average R²: {avg_r2:.3f}")
        print(f"Countries analyzed: {len(regression_results)}")
    else:
        print("No regression results generated.")