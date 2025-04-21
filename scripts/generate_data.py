import QuantLib as ql
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm

def generate_american_option_data(num_samples=100, grid_size=64):
    """
    Generate a dataset of American option scenarios and solutions
    
    Args:
        num_samples: Number of option scenarios to generate
        grid_size: Resolution of the solution grid
    
    Returns:
        inputs: Dictionary of input parameters
        solutions: Array of price surfaces
    """
    # Initialize containers
    inputs = {
        'strike': [],
        'spot': [],
        'rate': [],
        'dividend': [],
        'volatility': [],
        'time_to_expiry': []
    }
    
    # Output will be a grid of prices for different stock prices and times
    solutions = []
    
    # Today's date for option calculations
    today = ql.Date.todaysDate()
    
    # Day counter for calculating time factors
    day_counter = ql.Actual365Fixed()
    
    # Generate random option scenarios
    for _ in tqdm(range(num_samples), desc="Generating options", unit="sample"):
        # Random parameters (within reasonable ranges)
        strike = np.random.uniform(80.0, 120.0)
        spot = np.random.uniform(70.0, 130.0)
        rate = np.random.uniform(0.01, 0.08)
        dividend = np.random.uniform(0.0, 0.03)
        volatility = np.random.uniform(0.1, 0.5)
        days_to_expiry = np.random.randint(30, 365)
        
        # Store input parameters
        inputs['strike'].append(strike)
        inputs['spot'].append(spot)
        inputs['rate'].append(rate)
        inputs['dividend'].append(dividend)
        inputs['volatility'].append(volatility)
        inputs['time_to_expiry'].append(days_to_expiry/365.0)
        
        # Setup QuantLib option calculation
        expiry_date = today + ql.Period(days_to_expiry, ql.Days)
        
        # Create option
        payoff = ql.PlainVanillaPayoff(ql.Option.Put, strike)
        exercise = ql.AmericanExercise(today, expiry_date)
        option = ql.VanillaOption(payoff, exercise)
        
        # Create pricing engine
        spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
        flat_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(today, rate, day_counter))
        dividend_yield = ql.YieldTermStructureHandle(
            ql.FlatForward(today, dividend, day_counter))
        flat_vol_ts = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(today, ql.NullCalendar(), volatility, day_counter))
        
        bsm_process = ql.BlackScholesMertonProcess(spot_handle, 
                                                  dividend_yield, 
                                                  flat_ts, 
                                                  flat_vol_ts)
        
        # Using finite differences for American option
        engine = ql.FdBlackScholesVanillaEngine(bsm_process, 
                                                grid_size,  # time steps
                                                grid_size)  # space steps
        option.setPricingEngine(engine)
        
        # Create a price surface
        # Stock price from 50% to 150% of spot
        stock_prices = np.linspace(spot * 0.5, spot * 1.5, grid_size)
        # Time points from today to expiry
        time_points = np.linspace(0, days_to_expiry/365.0, grid_size)
        
        # Create the price surface
        price_surface = np.zeros((grid_size, grid_size))
        
        for i, s in enumerate(stock_prices):
            for j, t in enumerate(time_points):
                if t == 0:  # At time 0, use option payoff
                    price_surface[j, i] = max(0, strike - s) if payoff.optionType() == ql.Option.Put else max(0, s - strike)
                else:
                    # Calculate time to that point
                    time_date = today + ql.Period(int(t * 365), ql.Days)
                    if time_date > expiry_date:
                        time_date = expiry_date
                        
                    # Create a temp option with this expiry
                    temp_exercise = ql.AmericanExercise(today, time_date)
                    temp_option = ql.VanillaOption(payoff, temp_exercise)
                    
                    # Set up the process with this spot price
                    temp_spot = ql.QuoteHandle(ql.SimpleQuote(s))
                    temp_process = ql.BlackScholesMertonProcess(temp_spot, 
                                                              dividend_yield, 
                                                              flat_ts, 
                                                              flat_vol_ts)
                    
                    # Create engine and price
                    temp_engine = ql.FdBlackScholesVanillaEngine(temp_process, 100, 100)
                    temp_option.setPricingEngine(temp_engine)
                    
                    try:
                        price_surface[j, i] = temp_option.NPV()
                    except:
                        # Fallback to intrinsic value if pricing fails
                        price_surface[j, i] = max(0, strike - s) if payoff.optionType() == ql.Option.Put else max(0, s - strike)
        
        solutions.append(price_surface)
    
    # Convert inputs to numpy arrays
    for key in inputs:
        inputs[key] = np.array(inputs[key])
    
    # Stack all solutions
    solutions = np.array(solutions)
    
    return inputs, solutions

def plot_sample(inputs, solutions, index=0):
    """Plot a sample from the dataset"""
    fig = plt.figure(figsize=(12, 5))
    
    # Plot price surface
    ax1 = fig.add_subplot(121, projection='3d')
    grid_size = solutions.shape[1]
    X, Y = np.meshgrid(
        np.linspace(0.5, 1.5, grid_size),  # Normalized stock price
        np.linspace(0, 1, grid_size)       # Normalized time
    )
    surf = ax1.plot_surface(X, Y, solutions[index], cmap='viridis')
    ax1.set_xlabel('Stock Price / Spot')
    ax1.set_ylabel('Time to Expiry')
    ax1.set_zlabel('Option Price')
    ax1.set_title('American Option Price Surface')
    
    # Display parameters
    ax2 = fig.add_subplot(122)
    ax2.axis('off')
    parameter_text = f"""
    Strike: {inputs['strike'][index]:.2f}
    Spot: {inputs['spot'][index]:.2f}
    Risk-Free Rate: {inputs['rate'][index]:.2%}
    Dividend Yield: {inputs['dividend'][index]:.2%}
    Volatility: {inputs['volatility'][index]:.2%}
    Time to Expiry: {inputs['time_to_expiry'][index]:.2f} years
    """
    ax2.text(0.1, 0.5, parameter_text, fontsize=12)
    
    plt.tight_layout()
    return fig

def save_dataset(inputs, solutions, filename='american_option_dataset'):
    """Save the dataset to disk"""
    np.savez_compressed(
        filename,
        strike=inputs['strike'],
        spot=inputs['spot'],
        rate=inputs['rate'],
        dividend=inputs['dividend'],
        volatility=inputs['volatility'],
        time_to_expiry=inputs['time_to_expiry'],
        solutions=solutions
    )
    print(f"Dataset saved to {filename}.npz")

def load_dataset(filename='american_option_dataset'):
    """Load the dataset from disk"""
    data = np.load(f"{filename}.npz")
    inputs = {
        'strike': data['strike'],
        'spot': data['spot'],
        'rate': data['rate'],
        'dividend': data['dividend'],
        'volatility': data['volatility'],
        'time_to_expiry': data['time_to_expiry']
    }
    solutions = data['solutions']
    return inputs, solutions

if __name__ == "__main__":
    # Generate a dataset
    print("Generating American option dataset...")
    inputs, solutions = generate_american_option_data(num_samples=100, grid_size=64)
    
    # Plot a sample
    fig = plot_sample(inputs, solutions)
    plt.savefig('sample_american_option.png')
    plt.close(fig)
    
    # Save the dataset
    save_dataset(inputs, solutions)
    
    print(f"Generated dataset with {len(inputs['strike'])} samples")
    print(f"Input shape: {len(inputs)} parameters")
    print(f"Solutions shape: {solutions.shape}")