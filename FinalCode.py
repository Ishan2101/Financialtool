import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets

# Industry standard returns and volatilities for other asset classes
asset_classes = {
    "Mutual Funds": {"return": 0.13, "volatility": 0.10},
    "Bonds": {"return": 0.06, "volatility": 0.05},
    "Commodities": {"return": 0.06, "volatility": 0.15},
    "Real Estate": {"return": 0.05, "volatility": 0.08},
    "Fixed Deposit": {"return": 0.05, "volatility": 0.00},
}

# Function to fetch historical data from Yahoo Finance
def fetch_data(tickers):
    try:
        data = yf.download(tickers, start='2023-01-01', end='2024-01-01')['Adj Close']
        if data.empty or data.isnull().all().all():
            raise ValueError("Data retrieval failed or returned empty data for the tickers provided.")
        return data
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

# Function to perform Monte Carlo Simulation for stocks
def monte_carlo_simulation(initial_prices, mean_returns, cov_matrix, time_horizon=252, num_simulations=1000):
    num_assets = len(initial_prices)
    simulated_prices = np.zeros((time_horizon, num_assets, num_simulations))

    for sim in range(num_simulations):
        prices = np.zeros((time_horizon, num_assets))
        prices[0] = initial_prices
        for t in range(1, time_horizon):
            try:
                z = np.random.multivariate_normal(mean_returns, cov_matrix)
            except np.linalg.LinAlgError:
                print("Error generating random shocks due to invalid covariance matrix.")
                continue
            prices[t] = prices[t - 1] * np.exp(z)
        simulated_prices[:, :, sim] = prices

    return simulated_prices

# Function to calculate Value at Risk (VaR)
def calculate_var(simulated_prices, confidence_level=0.95):
    simulated_returns = np.log(simulated_prices[1:] / simulated_prices[:-1])
    portfolio_returns = simulated_returns.sum(axis=(1, 2))
    var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
    return var

# Function to classify risk based on VaR
def classify_risk(var):
    if var < -0.05:
        return "High Risk Portfolio", (
            "Your portfolio is classified as high risk. This indicates significant potential losses during market downturns.\n"
            "To manage risk better, consider diversifying across asset classes, reducing exposure to highly volatile assets, "
            "and reviewing your investment time horizon.\n")
    elif var < -0.02:
        return "Medium Risk Portfolio", (
            "Your portfolio is classified as medium risk. This suggests moderate potential losses with market fluctuations.\n"
            "To manage risk, consider balancing your portfolio with more stable assets like bonds or fixed deposits.\n")
    else:
        return "Low Risk Portfolio", (
            "Your portfolio is classified as low risk. This indicates a conservative approach with limited potential losses.\n"
            "To maintain this low-risk profile, continue focusing on stable, low-volatility assets and regularly review your allocation.\n")

# PyQt5 GUI Class
class FinancialModelApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Financial Model - Monte Carlo Simulation & VaR")
        self.setGeometry(100, 100, 800, 600)
        self.initUI()

    def initUI(self):
        layout = QtWidgets.QVBoxLayout()

        # Instructions for the user
        self.instructions = QtWidgets.QLabel("Select asset classes and enter the stock tickers for Monte Carlo Simulation (comma-separated).")
        layout.addWidget(self.instructions)

        # Asset class selection
        self.asset_listbox = QtWidgets.QListWidget()
        self.asset_listbox.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.asset_listbox.addItems(["Stocks", "Mutual Funds", "Bonds", "Commodities", "Real Estate", "Fixed Deposit"])
        layout.addWidget(self.asset_listbox)

        # Ticker input for stocks
        self.ticker_input = QtWidgets.QLineEdit(self)
        self.ticker_input.setPlaceholderText("Enter Stock Tickers (e.g., AAPL, MSFT) - Comma-separated")
        layout.addWidget(self.ticker_input)

        # Run simulation button
        self.run_button = QtWidgets.QPushButton("Run Simulation")
        self.run_button.clicked.connect(self.get_allocations)
        layout.addWidget(self.run_button)

        # Text area to display explanations and results
        self.results_display = QtWidgets.QTextEdit(self)
        self.results_display.setReadOnly(True)
        layout.addWidget(self.results_display)

        self.setLayout(layout)

    def get_allocations(self):
        selected_assets = [item.text() for item in self.asset_listbox.selectedItems()]

        if not selected_assets:
            self.results_display.setText("Please select at least one asset class.")
            return

        # Dialog for entering allocation percentages
        self.allocation_dialog = QtWidgets.QDialog(self)
        self.allocation_dialog.setWindowTitle("Enter Asset Allocation Percentages")
        dialog_layout = QtWidgets.QVBoxLayout()

        self.allocation_inputs = {}
        for asset in selected_assets:
            label = QtWidgets.QLabel(f"Enter percentage for {asset}:")
            input_field = QtWidgets.QLineEdit(self.allocation_dialog)
            input_field.setPlaceholderText("Enter percentage (e.g., 20 for 20%)")
            self.allocation_inputs[asset] = input_field
            dialog_layout.addWidget(label)
            dialog_layout.addWidget(input_field)

        submit_button = QtWidgets.QPushButton("Submit")
        submit_button.clicked.connect(self.validate_allocations)
        dialog_layout.addWidget(submit_button)

        self.allocation_dialog.setLayout(dialog_layout)
        self.allocation_dialog.exec_()

    def validate_allocations(self):
        selected_assets = [item.text() for item in self.asset_listbox.selectedItems()]
        total_allocation = 0
        allocations = {}

        # Validate and collect allocations
        try:
            for asset, input_field in self.allocation_inputs.items():
                allocation = float(input_field.text())
                if allocation < 0:
                    self.results_display.setText(f"Allocation for {asset} cannot be negative.")
                    return
                allocations[asset] = allocation / 100  # Convert to a decimal
                total_allocation += allocation

            # Check if total allocation equals 100%
            if total_allocation != 100:
                self.results_display.setText("Total allocation must equal 100%. Please adjust your percentages.")
                return

            self.run_simulation(selected_assets, allocations)

        except ValueError:
            self.results_display.setText("Invalid input detected. Please enter valid percentages.")
            return

    def run_simulation(self, selected_assets, allocations):
        tickers = self.ticker_input.text().strip().split(',')
        tickers = [ticker.strip() for ticker in tickers if ticker.strip()]  # Clean ticker input
        explanation = "Educational Insights:\n"

        try:
            # Initialize portfolio return and volatility
            portfolio_return = 0.0
            portfolio_volatility = 0.0

            # Handle Monte Carlo Simulations if Stocks are included
            if "Stocks" in selected_assets and tickers:
                data = fetch_data(tickers)
                if data.empty:
                    self.results_display.setText("Failed to retrieve data for the selected tickers. Please check the tickers and try again.")
                    return

                initial_prices = data.iloc[-1].values
                returns = np.log(data / data.shift(1)).dropna()

                if returns.empty or returns.isnull().all().all():
                    self.results_display.setText("No valid return data found. Unable to perform simulations.")
                    return

                mean_returns = returns.mean().values
                cov_matrix = returns.cov().values

                # Perform Monte Carlo Simulation
                simulated_prices = monte_carlo_simulation(initial_prices, mean_returns, cov_matrix)

                # Calculating the weighted returns and volatility for stocks
                stock_weights = allocations["Stocks"]
                weighted_return = float(np.dot(stock_weights, mean_returns) * 252)
                weighted_volatility = float(np.sqrt(np.dot(stock_weights, np.dot(cov_matrix, stock_weights))) * np.sqrt(252))

                portfolio_return += weighted_return
                portfolio_volatility += weighted_volatility

                # Plotting Monte Carlo Simulations for each stock individually
                for i, ticker in enumerate(tickers):
                    plt.figure(figsize=(12, 6))
                    for j in range(min(100, simulated_prices.shape[2])):  # Plot up to 100 simulations
                        plt.plot(simulated_prices[:, i, j], lw=0.5, alpha=0.6)
                    plt.title(f'Monte Carlo Simulations of {ticker} Stock Price Over 1 Year')
                    plt.xlabel('Days')
                    plt.ylabel('Simulated Prices')
                    plt.grid(True)
                    plt.show()

                    # Plotting histogram of simulated final prices for each stock
                    plt.figure(figsize=(10, 6))
                    plt.hist(simulated_prices[-1, i, :], bins=50, alpha=0.7, color='blue')
                    plt.title(f'Distribution of Simulated Final Prices for {ticker}')
                    plt.xlabel('Final Simulated Prices')
                    plt.ylabel('Frequency')
                    plt.grid(True)
                    plt.show()

                # Calculate Value at Risk (VaR)
                var = calculate_var(simulated_prices)
                risk_level, advice = classify_risk(var)
                explanation += f"\nValue-at-Risk (VaR) Calculation:\n"
                explanation += "VaR measures the potential loss in value of your portfolio over a specified time period with a given confidence level.\n"
                explanation += f"The VaR of your portfolio is {var:.2f}, which indicates the potential downside risk.\n"
                explanation += f"\nRisk Level: {risk_level}\n"
                explanation += advice

            # Include other asset classes in the portfolio calculations
            for asset in selected_assets:
                if asset in asset_classes and asset != "Stocks":
                    portfolio_return += asset_classes[asset]["return"] * allocations[asset]
                    portfolio_volatility += asset_classes[asset]["volatility"] * allocations[asset]
                    explanation += f"Including {asset}, which has an annual expected return of {asset_classes[asset]['return'] * 100:.2f}% and volatility of {asset_classes[asset]['volatility'] * 100:.2f}%, further diversifies your portfolio, potentially balancing the risk.\n"

            # Display final expected return and volatility
            explanation += f"\nThe overall expected return of your portfolio is {portfolio_return * 100:.2f}% per year."
            explanation += f"\nThe overall portfolio volatility is estimated to be {portfolio_volatility * 100:.2f}% per year.\n"

            # Update display with the full explanation
            self.results_display.setText(explanation)

        except Exception as e:
            self.results_display.setText(f"An error occurred: {str(e)}")

# Main execution
if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ex = FinancialModelApp()
    ex.show()
    sys.exit(app.exec_())
