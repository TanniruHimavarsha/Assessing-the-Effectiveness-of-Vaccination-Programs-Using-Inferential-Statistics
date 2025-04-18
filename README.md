# Vaccination Program Effectiveness Analysis

This project analyzes the effectiveness of vaccination programs using inferential statistics. It performs hypothesis testing and generates confidence intervals to assess the impact of vaccination campaigns on infection rates.

## Features

- Data loading and preprocessing
- Statistical analysis including:
  - Hypothesis testing
  - Confidence interval calculation
  - Correlation analysis
- Visualization of results
- Automated report generation

## Requirements

- Python 3.8+
- Required packages listed in `requirements.txt`

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Basic usage with sample data:
   ```bash
   python vaccination_analysis.py
   ```

2. Using your own data:
   ```python
   from vaccination_analysis import VaccinationAnalysis
   
   analysis = VaccinationAnalysis('path_to_your_data.csv')
   analysis.visualize_results()
   report = analysis.generate_report()
   print(report)
   ```

## Data Format

The program expects a CSV file with the following columns:
- `vaccination_rate`: Percentage of vaccinated population (0-1)
- `infection_rate`: Infection rate in the population (0-1)
- `region`: Geographic region or area name

## Output

The program generates:
1. Statistical analysis results
2. Visualizations saved as 'vaccination_analysis_results.png'
3. A detailed report saved as 'vaccination_analysis_report.txt'

## Statistical Methods

The analysis includes:
- Pearson correlation between vaccination and infection rates
- Hypothesis testing with configurable significance level
- Confidence intervals for vaccination rates
- Regional comparisons

## License

MIT License 