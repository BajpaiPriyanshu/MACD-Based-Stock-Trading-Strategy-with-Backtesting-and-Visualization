import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def generate_sample_data(days=252):
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    returns = np.random.normal(0.001, 0.02, days)
    prices = [100]
    for i in range(1, days):
        prices.append(prices[-1] * (1 + returns[i]))
    data = pd.DataFrame({
        'Date': dates,
        'Close': prices
    })
    data.set_index('Date', inplace=True)
    return data

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    ema_fast = data['Close'].ewm(span=fast_period).mean()
    ema_slow = data['Close'].ewm(span=slow_period).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period).mean()
    histogram = macd_line - signal_line
    data['EMA_Fast'] = ema_fast
    data['EMA_Slow'] = ema_slow
    data['MACD'] = macd_line
    data['Signal'] = signal_line
    data['Histogram'] = histogram
    return data

def generate_signals(data):
    data['Position'] = 0
    data['Signal_Flag'] = 0
    for i in range(1, len(data)):
        if data['MACD'].iloc[i] > data['Signal'].iloc[i] and data['MACD'].iloc[i-1] <= data['Signal'].iloc[i-1]:
            data['Signal_Flag'].iloc[i] = 1
            data['Position'].iloc[i] = 1
        elif data['MACD'].iloc[i] < data['Signal'].iloc[i] and data['MACD'].iloc[i-1] >= data['Signal'].iloc[i-1]:
            data['Signal_Flag'].iloc[i] = -1
            data['Position'].iloc[i] = -1
        else:
            data['Position'].iloc[i] = data['Position'].iloc[i-1]
    return data

def backtest_strategy(data, initial_capital=10000):
    data['Returns'] = data['Close'].pct_change()
    data['Strategy_Returns'] = data['Position'].shift(1) * data['Returns']
    data['Cumulative_Market'] = (1 + data['Returns']).cumprod() * initial_capital
    data['Cumulative_Strategy'] = (1 + data['Strategy_Returns']).cumprod() * initial_capital
    total_return_market = (data['Cumulative_Market'].iloc[-1] / initial_capital - 1) * 100
    total_return_strategy = (data['Cumulative_Strategy'].iloc[-1] / initial_capital - 1) * 100
    print("=== BACKTEST RESULTS ===")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Market Value: ${data['Cumulative_Market'].iloc[-1]:,.2f}")
    print(f"Final Strategy Value: ${data['Cumulative_Strategy'].iloc[-1]:,.2f}")
    print(f"Buy & Hold Return: {total_return_market:.2f}%")
    print(f"MACD Strategy Return: {total_return_strategy:.2f}%")
    print(f"Strategy Outperformance: {total_return_strategy - total_return_market:.2f}%")
    return data

def plot_macd_analysis(data):
    fig1, ax1 = plt.subplots(figsize=(15, 6))
    ax1.plot(data.index, data['Close'], label='Close Price', color='black', linewidth=2)
    ax1.plot(data.index, data['EMA_Fast'], label='EMA Fast (12)', color='blue', alpha=0.7)
    ax1.plot(data.index, data['EMA_Slow'], label='EMA Slow (26)', color='red', alpha=0.7)

    buy_signals = data[data['Signal_Flag'] == 1]
    sell_signals = data[data['Signal_Flag'] == -1]

    ax1.scatter(buy_signals.index, buy_signals['Close'],
                color='green', marker='^', s=100, label='Buy Signal')
    ax1.scatter(sell_signals.index, sell_signals['Close'],
                color='red', marker='v', s=100, label='Sell Signal')

    ax1.set_title('Stock Price with MACD Signals')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    fig2, ax2 = plt.subplots(figsize=(15, 6))
    ax2.plot(data.index, data['MACD'], label='MACD Line', color='blue', linewidth=2)
    ax2.plot(data.index, data['Signal'], label='Signal Line', color='red', linewidth=2)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)

    ax2.scatter(buy_signals.index, buy_signals['MACD'],
                color='green', marker='^', s=60, alpha=0.8)
    ax2.scatter(sell_signals.index, sell_signals['MACD'],
                color='red', marker='v', s=60, alpha=0.8)

    ax2.set_title('MACD and Signal Lines')
    ax2.set_ylabel('MACD Value')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    fig3, ax3 = plt.subplots(figsize=(15, 4))
    colors = ['green' if x > 0 else 'red' for x in data['Histogram']]
    ax3.bar(data.index, data['Histogram'], color=colors, alpha=0.6, width=1)

    ax3.set_title('MACD Histogram')
    ax3.set_ylabel('Value')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    fig4, ax4 = plt.subplots(figsize=(15, 6))
    ax4.plot(data.index, data['Cumulative_Market'],
             label='Buy & Hold Strategy', color='blue', linewidth=2)
    ax4.plot(data.index, data['Cumulative_Strategy'],
             label='MACD Strategy', color='green', linewidth=2)

    ax4.set_title('Strategy Performance Comparison')
    ax4.set_ylabel('Portfolio Value ($)')
    ax4.set_xlabel('Date')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    plt.show()

def main():
    print("=== MACD INDICATOR ANALYSIS ===")
    print("Generating sample data...")
    data = generate_sample_data(days=252)
    print(f"Data loaded: {len(data)} days")
    print("Calculating MACD indicator...")
    data = calculate_macd(data)
    print("Generating buy/sell signals...")
    data = generate_signals(data)
    print("Running backtest...")
    data = backtest_strategy(data)
    print("\n=== SAMPLE DATA ===")
    print(data[['Close', 'MACD', 'Signal', 'Histogram', 'Signal_Flag']].head(30))
    buy_count = len(data[data['Signal_Flag'] == 1])
    sell_count = len(data[data['Signal_Flag'] == -1])
    print(f"\nTotal Buy Signals: {buy_count}")
    print(f"Total Sell Signals: {sell_count}")
    print("\nGenerating plots...")
    plot_macd_analysis(data)
    return data

if __name__ == "__main__":
    result_data = main()
    print("\n=== ANALYSIS COMPLETE ===")
    print("The MACD indicator helps identify trend changes and momentum shifts.")
    print("Key Points:")
    print("• Buy when MACD line crosses ABOVE signal line")
    print("• Sell when MACD line crosses BELOW signal line")
    print("• Histogram shows the strength of the trend")
    print("• This is a basic strategy - real trading requires more sophisticated risk management!")
