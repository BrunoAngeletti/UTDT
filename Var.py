import yfinance as yf
import numpy as np
import pandas as pd

class VaRCalculator:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = self._download_data()
        self.returns = self._calculate_returns()

    def _download_data(self):
        asset = yf.download(self.ticker, start=self.start_date, end=self.end_date,auto_adjust=False, interval='1d')
        if asset.empty:
            raise ValueError(f"No se encontraron datos para {self.ticker} en el periodo dado.")
        return asset['Adj Close']

    def _calculate_returns(self):
        returns = self.data.pct_change().dropna()
        if returns.empty:
            raise ValueError("Serie de retornos vacía. Ajusta el período o activo seleccionado.")
        return returns

    def var_parametric(self, confidence_level=0.05):
        mean = self.returns.mean()
        std_dev = self.returns.std()
        var = mean + std_dev * np.percentile(np.random.normal(0, 1, 100000), confidence_level * 100)
        return abs(var)

    def var_historical(self, confidence_level=0.05):
        var = np.percentile(self.returns, confidence_level * 100)
        return abs(var)

    def var_monte_carlo(self, confidence_level=0.05, simulations=10000):
        mean = self.returns.mean()
        std_dev = self.returns.std()
        simulated_returns = np.random.normal(mean, std_dev, simulations)
        var = np.percentile(simulated_returns, confidence_level * 100)
        return abs(var)

    def get_var_summary(self, confidence_levels=[0.01, 0.05, 0.1]):
        summary = []
        for cl in confidence_levels:
            summary.append({
                'Confidence Level': f"{int(cl*100)}%",
                'Parametric VaR': self.var_parametric(cl).values[0],  # Solo esta lo necesita
                'Historical VaR': self.var_historical(cl),
                'Monte Carlo VaR': self.var_monte_carlo(cl)
            })
        return pd.DataFrame(summary)



# Ejemplo de uso (asegúrate de seleccionar fechas con datos):
if __name__ == "__main__":
    ticker = 'SPY'
    start_date = '2024-01-01'
    end_date = '2025-01-01'

    try:
        var_calc = VaRCalculator(ticker, start_date, end_date)
        summary = var_calc.get_var_summary()
        print(summary)
    except ValueError as e:
        print(e)
