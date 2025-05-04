import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, skew, kurtosis, gaussian_kde

class DistributionPlot:
    """
    Demonstrates:
      - Using a KDE-based mode for continuous data (so the mode isn't always 0).
      - Using the same bin edges for full, left-tail, and right-tail plots,
        avoiding loss of data.
    """

    def __init__(
        self,
        ticker='SPY',
        start_date='2000-01-01',
        # Bin, xlim, ylim for each subplot
        num_bins=500,
        full_xlim=(-0.05, 0.05),
        full_ylim=(0, 200),
        left_xlim=(-0.15, -0.05),
        left_ylim=(0, 6),
        right_xlim=(0.05, 0.15),
        right_ylim=(0, 6)
    ):
        self.ticker = ticker
        self.start_date = start_date
        self.df = None

        # Stats placeholders
        self.mean_return = None
        self.std_return = None
        self.skewness = None
        self.excess_kurtosis = None
        self.mode_val = None  # We'll store the *KDE-based* mode here

        # Plot customization
        self.num_bins = num_bins

        self.full_xlim = full_xlim
        self.full_ylim = full_ylim

        self.left_xlim = left_xlim
        self.left_ylim = left_ylim

        self.right_xlim = right_xlim
        self.right_ylim = right_ylim

        # Internal
        self.all_min = None
        self.all_max = None
        self.bin_edges = None
        self.xvals = None
        self.pdf_normal_scaled = None
        self.pdf_kde_scaled = None

    def download_data(self):
        """Download data via yfinance and compute daily log returns."""
        self.df = yf.download(self.ticker, start=self.start_date)
        self.df.dropna(inplace=True)

        self.df['Log_Return'] = np.log(
            self.df['Adj Close'] / self.df['Adj Close'].shift(1)
        )
        self.df.dropna(subset=['Log_Return'], inplace=True)

    def compute_statistics(self):
        """Compute mean, std, skewness, kurtosis. Then find KDE-based mode."""
        log_ret = self.df['Log_Return']
        self.mean_return = log_ret.mean()
        self.std_return = log_ret.std()
        self.skewness = skew(log_ret, bias=False)
        self.excess_kurtosis = kurtosis(log_ret, fisher=True)

        # ---- KDE-based mode calculation ----
        data = log_ret.values
        self.all_min, self.all_max = data.min(), data.max()

        # Evaluate KDE on a fine grid
        x_kde = np.linspace(self.all_min, self.all_max, 2000)
        kde = gaussian_kde(data)
        pdf_kde = kde(x_kde)

        # The "mode" is where the KDE is maximum
        idx_mode = np.argmax(pdf_kde)
        self.mode_val = x_kde[idx_mode]

    def prepare_plot_data(self):
        """
        Create bin edges for the *full range*, and compute scaled PDFs
        for normal & KDE.
        """
        data = self.df['Log_Return'].values

        # One set of bin edges for the entire range (no data missing)
        self.bin_edges = np.linspace(self.all_min, self.all_max, self.num_bins + 1)

        # Normal PDF & KDE for overlay
        self.xvals = np.linspace(self.all_min, self.all_max, 2000)
        pdf_normal = norm.pdf(self.xvals, self.mean_return, self.std_return)

        kde = gaussian_kde(data)
        pdf_kde = kde(self.xvals)

        # Scale both to match histogram raw counts
        total_points = len(data)
        bin_width = (self.all_max - self.all_min) / self.num_bins
        self.pdf_normal_scaled = pdf_normal * total_points * bin_width
        self.pdf_kde_scaled = pdf_kde * total_points * bin_width

    def plot_distribution(self):
        """
        Full distribution, plus two tail subplots that just "zoom" via xlim
        (so no data is discarded).
        """
        data = self.df['Log_Return'].values
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(18, 6), sharey=False)

        # ================================
        # (A) FULL DISTRIBUTION
        # ================================
        ax1.hist(
            data,
            bins=self.bin_edges,
            alpha=0.6,
            color='blue',
            label='Histogram (counts)'
        )
        ax1.plot(self.xvals, self.pdf_normal_scaled, color='orange', label='Normal PDF (scaled)')
        ax1.plot(self.xvals, self.pdf_kde_scaled, color='green', label='KDE (scaled)')

        # Vertical lines for mean & KDE-based mode
        ax1.axvline(self.mean_return, color='black', linestyle=':', linewidth=1, label='Mean')
        ax1.axvline(self.mode_val, color='red', linestyle='-', linewidth=1, label='Mode (KDE)')

        ax1.set_title('Full Distribution')
        ax1.set_xlabel('Log Return')
        ax1.set_ylabel('Number of Observations')
        ax1.legend()

        # Use user-specified axis limits
        ax1.set_xlim(self.full_xlim)
        ax1.set_ylim(self.full_ylim)

        # ================================
        # (B) LEFT TAIL (Zoom)
        # ================================
        # Plot the same data, same bins, but just zoom in on x
        ax2.hist(
            data,
            bins=self.bin_edges,
            alpha=0.6,
            color='blue'
        )
        ax2.plot(self.xvals, self.pdf_normal_scaled, color='orange')
        ax2.plot(self.xvals, self.pdf_kde_scaled, color='green')

        ax2.set_title('Left Tail (Zoom, Raw Counts)')
        ax2.set_xlabel('Log Return')
        ax2.set_ylabel('Number of Observations')

        ax2.set_xlim(self.left_xlim)
        ax2.set_ylim(self.left_ylim)

        # ================================
        # (C) RIGHT TAIL (Zoom)
        # ================================
        ax3.hist(
            data,
            bins=self.bin_edges,
            alpha=0.6,
            color='blue'
        )
        ax3.plot(self.xvals, self.pdf_normal_scaled, color='orange')
        ax3.plot(self.xvals, self.pdf_kde_scaled, color='green')

        ax3.set_title('Right Tail (Zoom, Raw Counts)')
        ax3.set_xlabel('Log Return')
        ax3.set_ylabel('Number of Observations')

        ax3.set_xlim(self.right_xlim)
        ax3.set_ylim(self.right_ylim)

        # Final formatting
        end_date_actual = self.df.index[-1].strftime('%Y-%m-%d')
        fig.suptitle(
            f"{self.ticker} Daily Log Returns (Raw Counts)\n"
            f"Start: {self.start_date}, End: {end_date_actual}\n"
            f"Mean: {self.mean_return:.6f}, Std: {self.std_return:.6f}\n"
            f"Mode: {self.mode_val:.6f}  (Skew: {self.skewness:.6f}, Kurt: {self.excess_kurtosis:.6f})",
            fontsize=14
        )

        plt.tight_layout()
        plt.show()

    def run(self):
        """Execute all steps."""
        self.download_data()
        self.compute_statistics()
        self.prepare_plot_data()
        self.plot_distribution()

# ----------------- EXAMPLE USAGE -------------------
if __name__ == "__main__":
    dp = DistributionPlot(
        ticker='',
        start_date='2000-01-01',
        num_bins=500,             # Bins for the full distribution
        full_xlim=(-0.05, 0.05),
        full_ylim=(0, 200),
        left_xlim=(-0.15, -0.05),
        left_ylim=(0, 6),
        right_xlim=(0.05, 0.15),
        right_ylim=(0, 6)
    )
    dp.run()

