import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportion_confint
import requests
from typing import Tuple, Dict, List
import json

class VaccinationAnalysis:
    def __init__(self, data_path: str = None):
        """
        Initialize the VaccinationAnalysis class.
        Args:
            data_path (str, optional): Path to the vaccination data file.
        """
        self.data = None
        if data_path:
            self.load_data(data_path)

    def load_data(self, data_path: str) -> None:
        """
        Load vaccination and infection data from a CSV file.
        Args:
            data_path (str): Path to the data file
        """
        self.data = pd.read_csv(data_path)
        print(f"Loaded data with shape: {self.data.shape}")

    def fetch_sample_data(self) -> None:
        """
        Fetch sample vaccination data from a public health API
        Note: Replace with actual API endpoint in production
        """
        # Sample data generation for demonstration
        np.random.seed(42)
        n_samples = 1000
        
        self.data = pd.DataFrame({
            'vaccination_rate': np.random.normal(0.7, 0.1, n_samples),
            'infection_rate': np.random.normal(0.1, 0.05, n_samples),
            'region': [f'Region_{i%5}' for i in range(n_samples)]
        })
        
        # Ensure realistic bounds
        self.data['vaccination_rate'] = self.data['vaccination_rate'].clip(0, 1)
        self.data['infection_rate'] = self.data['infection_rate'].clip(0, 1)

    def perform_hypothesis_test(self, alpha: float = 0.05) -> Dict:
        """
        Perform hypothesis testing to assess vaccination program effectiveness.
        
        H0: There is no correlation between vaccination rates and infection rates
        H1: There is a significant correlation between vaccination rates and infection rates
        
        Args:
            alpha (float): Significance level
            
        Returns:
            Dict: Results of the statistical tests
        """
        # Pearson correlation test
        correlation, p_value = stats.pearsonr(
            self.data['vaccination_rate'],
            self.data['infection_rate']
        )
        
        # Calculate confidence intervals for vaccination rate
        vax_mean = self.data['vaccination_rate'].mean()
        confidence = 0.95  # 95% confidence interval
        vax_ci = stats.t.interval(
            confidence=confidence,
            df=len(self.data)-1,
            loc=vax_mean,
            scale=stats.sem(self.data['vaccination_rate'])
        )
        
        return {
            'correlation': correlation,
            'p_value': p_value,
            'significant': p_value < alpha,
            'vaccination_rate_mean': vax_mean,
            'vaccination_rate_ci': vax_ci
        }

    def visualize_results(self) -> None:
        """
        Create visualizations of the vaccination program analysis.
        """
        plt.figure(figsize=(12, 8))
        
        # Scatter plot with regression line
        plt.subplot(2, 2, 1)
        sns.regplot(
            data=self.data,
            x='vaccination_rate',
            y='infection_rate',
            scatter_kws={'alpha': 0.5}
        )
        plt.title('Vaccination Rate vs Infection Rate')
        
        # Distribution of vaccination rates
        plt.subplot(2, 2, 2)
        sns.histplot(self.data['vaccination_rate'], kde=True)
        plt.title('Distribution of Vaccination Rates')
        
        # Box plot by region
        plt.subplot(2, 2, 3)
        sns.boxplot(data=self.data, x='region', y='vaccination_rate')
        plt.title('Vaccination Rates by Region')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('vaccination_analysis_results.png')
        plt.close()

    def generate_report(self) -> str:
        """
        Generate a summary report of the analysis.
        
        Returns:
            str: Summary report
        """
        results = self.perform_hypothesis_test()
        
        report = [
            "Vaccination Program Effectiveness Analysis",
            "=" * 40,
            f"\nSample Size: {len(self.data)}",
            f"\nKey Findings:",
            f"- Average Vaccination Rate: {results['vaccination_rate_mean']:.2%}",
            f"- Vaccination Rate 95% CI: ({results['vaccination_rate_ci'][0]:.2%}, {results['vaccination_rate_ci'][1]:.2%})",
            f"- Correlation with Infection Rate: {results['correlation']:.3f}",
            f"- Statistical Significance: {'Yes' if results['significant'] else 'No'} (p={results['p_value']:.4f})",
            "\nRecommendations:",
            "1. " + self._generate_recommendations(results),
        ]
        
        return "\n".join(report)

    def _generate_recommendations(self, results: Dict) -> str:
        """
        Generate recommendations based on analysis results.
        
        Args:
            results (Dict): Statistical analysis results
            
        Returns:
            str: Recommendations
        """
        if results['significant']:
            if results['correlation'] < 0:
                return ("The vaccination program shows significant effectiveness in reducing infection rates. "
                       "Continue current strategies and consider expanding coverage.")
            else:
                return ("Despite high vaccination rates, infection rates remain concerning. "
                       "Investigate vaccine effectiveness and potential new variants.")
        else:
            return ("Results are inconclusive. More data collection and analysis is recommended.")

def main():
    # Initialize analysis
    analysis = VaccinationAnalysis()
    
    # Load sample data
    analysis.fetch_sample_data()
    
    # Perform analysis and generate visualizations
    analysis.visualize_results()
    
    # Generate and print report
    report = analysis.generate_report()
    print(report)
    
    # Save report to file
    with open('vaccination_analysis_report.txt', 'w') as f:
        f.write(report)

if __name__ == "__main__":
    main() 