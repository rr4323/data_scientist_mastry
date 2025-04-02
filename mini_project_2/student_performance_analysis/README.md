# Student Performance Analysis System

A comprehensive data analysis system for evaluating student performance across multiple assessment components. This system provides detailed insights into student performance, attendance patterns, and overall academic progress.

## Features

- **Data Processing and Validation**
  - Robust data cleaning and validation
  - Handling of missing values and outliers
  - Standardized name normalization
  - Attendance tracking and validation

- **Performance Analysis**
  - Individual test performance analysis
  - Weighted percentage calculations
  - Performance categorization (Excellent, Good, Average, Needs Improvement)
  - Attendance impact analysis
  - Statistical analysis of test scores

- **Visualization**
  - Top performers analysis
  - Performance distribution charts
  - Attendance vs. Performance correlation
  - Test score distributions
  - Box plots for outlier detection
  - Correlation heatmaps

- **Output Generation**
  - Excel report with multiple sheets
  - High-quality visualizations saved as PNG files
  - Detailed statistical summaries
  - Outlier analysis reports

## Project Structure

```
advance_data_scientist/
└── mini_project_2/
    └── student_performance_analysis/
        ├── student_analysis.py      # Main analysis script
        ├── student.xlsx            # Input data file
        ├── student_analysis_report.md  # Detailed analysis report
        ├── plots/                  # Generated visualizations
        │   ├── main_analysis.png
        │   ├── performance_trends.png
        │   ├── correlation_matrix.png
        │   ├── score_distributions.png
        │   ├── box_plots.png
        │   └── combined_box_plots.png
        └── student_analysis_results.xlsx  # Generated analysis report
```

## Input Data Format

The system expects an Excel file (`student.xlsx`) with two sheets:

1. **Marks Sheet**
   - Student names
   - Mini Test 1 scores (max 20)
   - Mini Test 2 scores (max 20)
   - Live Test scores (max 40)
   - Assignment scores (max 20)

2. **Attendance Sheet**
   - Student names
   - Attendance records for each session

## Performance Calculation

The system calculates performance using the following weights:
- Mini Test 1: 20%
- Mini Test 2: 20%
- Live Test: 40%
- Assignment: 20%

Performance Categories:
- Excellent: ≥ 85%
- Good: ≥ 71%
- Average: ≥ 50%
- Needs Improvement: < 50%

## Requirements

- Python 3.10 or higher
- Required packages:
  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scipy
  - scikit-learn
  - openpyxl (for Excel file handling)

### Linux-Specific Requirements

For Linux users, additional system packages are required for GUI support:

```bash
sudo apt-get update
sudo apt-get install python3-tk python3-qt5
pip install tk PyQt5
```

## Installation

1. Navigate to the project directory:
```bash
cd advance_data_scientist/mini_project_2/student_performance_analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv student
source student/bin/activate  # On Linux/Mac
# or
student\Scripts\activate  # On Windows
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

4. For Linux users, install additional system packages:
```bash
sudo apt-get update
sudo apt-get install python3-tk python3-qt5
pip install tk PyQt5
```

## Usage

1. Prepare your input data in the required Excel format
2. Run the analysis script:
```bash
python student_analysis.py
```

3. Check the generated outputs:
   - `plots/` directory for visualizations
   - `student_analysis_results.xlsx` for detailed analysis

## Output Files

1. **Visualizations** (`plots/` directory):
   - `main_analysis.png`: Overview of key metrics
   - `performance_trends.png`: Performance trends across tests
   - `correlation_matrix.png`: Correlation between different metrics
   - `score_distributions.png`: Distribution of scores for each test
   - `box_plots.png`: Detailed box plots for each test
   - `combined_box_plots.png`: Combined view of all test distributions

2. **Analysis Reports**:
   - `student_analysis_report.md`: Detailed markdown report with analysis findings and insights
   - `student_analysis_results.xlsx`: Excel report with multiple sheets:
     - Complete Data: Raw and processed data
     - Statistics: Basic statistical measures
     - Performance Categories: Distribution of performance levels
     - Attendance Analysis: Attendance patterns and impact
     - Correlations: Correlation matrix

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.