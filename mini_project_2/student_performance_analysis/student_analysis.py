import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
import re
from typing import Dict, List, Union, Any
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration and Constants
# =============================================================================
# Define maximum marks for each component (total = 100)
MAX_MARKS_CONFIG = {
    'Mini Test 1': 20,    # 20% of total marks
    'Mini Test 2': 20,    # 20% of total marks
    'Live Test': 40,      # 40% of total marks
    'Assignment': 20      # 20% of total marks
}

# Performance classification thresholds
PERFORMANCE_THRESHOLDS = {
    'Excellent': 85,
    'Good': 71,
    'Average': 50
}

# Visualization settings
PLOT_STYLE = {
    'figsize': (15, 10),
    'dpi': 300,
    'style': 'whitegrid',
    'palette': 'husl'
}

# =============================================================================
# Data Validation and Cleaning Classes
# =============================================================================
class DataTypeValidator:
    """Class to handle data type validation and conversion"""
    
    @staticmethod
    def validate_numeric(value: Any) -> Union[float, None]:
        """Validate and convert numeric values"""
        if pd.isna(value):
            return None
        try:
            if isinstance(value, str):
                value = re.sub(r'[^\d.-]', '', value)
            return float(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def validate_string(value: Any) -> str:
        """Validate and clean string values"""
        if pd.isna(value):
            return ""
        return str(value).strip()

    @staticmethod
    def validate_attendance(value: Any) -> int:
        """Validate and convert attendance markers"""
        if pd.isna(value):
            return 0
        value = str(value).strip().upper()
        positive_markers = {'Y', 'YES', 'PRESENT', '1', 'P', 'TRUE'}
        negative_markers = {'N', 'NO', 'ABSENT', '0', 'A', 'FALSE'}
        
        if value in positive_markers:
            return 1
        elif value in negative_markers:
            return 0
        else:
            print(f"Warning: Unexpected attendance value '{value}' found. Defaulting to 0.")
            return 0

# =============================================================================
# Data Processing Functions
# =============================================================================
def validate_and_clean_data(df: pd.DataFrame, column_types: Dict[str, str]) -> pd.DataFrame:
    """Validate and clean DataFrame based on specified column types"""
    validator = DataTypeValidator()
    
    for column, dtype in column_types.items():
        if column not in df.columns:
            print(f"Warning: Column '{column}' not found in DataFrame")
            continue
            
        if dtype == 'numeric':
            df[column] = df[column].apply(validator.validate_numeric)
            df[column] = pd.to_numeric(df[column], errors='coerce')
        elif dtype == 'string':
            df[column] = df[column].apply(validator.validate_string)
        elif dtype == 'attendance':
            df[column] = df[column].apply(validator.validate_attendance)
            
    return df

def validate_marks(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Validate marks are within acceptable range"""
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        mask = (df[col] < 0) | (df[col] > 100)
        if mask.any():
            print(f"Warning: Found invalid marks in {col}")
            valid_median = df[~mask][col].median()
            df.loc[mask, col] = valid_median
    return df

def detect_outliers(df: pd.DataFrame, columns: List[str], threshold: float = 3) -> pd.DataFrame:
    """Detect and handle outliers using Z-score method"""
    for col in columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Warning: Column {col} is not numeric. Converting to numeric...")
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        z_scores = np.abs(stats.zscore(df[col]))
        outliers = z_scores > threshold
        if outliers.any():
            print(f"Found {outliers.sum()} outliers in {col}")
            df.loc[outliers, col] = df[col].median()
    return df

def normalize_names(name: str) -> str:
    """Advanced name normalization"""
    if pd.isna(name):
        return ""
    name = ' '.join(str(name).split())
    name = name.title()
    name = re.sub(r'[^a-zA-Z\s\'-]', '', name)
    name = re.sub(r'\s+', ' ', name)
    name = re.sub(r'-+', '-', name)
    return name.strip()

def scale_marks_to_100(value: float, max_marks: float) -> float:
    """Scale marks to 100-point scale"""
    return (value / max_marks) * 100

def classify_performance(percentage: float) -> str:
    """Classify student performance based on percentage"""
    if percentage >= PERFORMANCE_THRESHOLDS['Excellent']:
        return 'Excellent'
    elif percentage >= PERFORMANCE_THRESHOLDS['Good']:
        return 'Good'
    elif percentage >= PERFORMANCE_THRESHOLDS['Average']:
        return 'Average'
    else:
        return 'Needs Improvement'

# =============================================================================
# Data Loading and Initial Processing
# =============================================================================
def load_data():
    """Load and validate input data"""
    try:
        marks_df = pd.read_excel('student.xlsx', sheet_name='Marks')
        attendance_df = pd.read_excel('student.xlsx', sheet_name='Attendance')
        return marks_df, attendance_df
    except FileNotFoundError:
        print("Error: 'student.xlsx' file not found!")
        exit(1)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        exit(1)

def preprocess_data(marks_df: pd.DataFrame, attendance_df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess and merge the datasets"""
    print("\n=== Starting Advanced Data Preprocessing ===")
    
    # Define column types
    marks_column_types = {
        'Name': 'string',
        'Mini Test 1': 'numeric',
        'Mini Test 2': 'numeric',
        'Live Test': 'numeric',
        'Assignment': 'numeric'
    }

    attendance_column_types = {
        'Name': 'string'
    }
    attendance_column_types.update({
        col: 'attendance' for col in attendance_df.columns 
        if col.startswith('Attendance') or col != 'Name'
    })

    # Validate and clean data
    marks_df = validate_and_clean_data(marks_df, marks_column_types)
    attendance_df = validate_and_clean_data(attendance_df, attendance_column_types)

    # Clean names
    marks_df['Name'] = marks_df['Name'].apply(normalize_names)
    attendance_df['Name'] = attendance_df['Name'].apply(normalize_names)

    # Process marks
    marks_columns = list(MAX_MARKS_CONFIG.keys())
    marks_df = validate_marks(marks_df, marks_columns)
    marks_df = detect_outliers(marks_df, marks_columns)

    # Process attendance
    attendance_columns = [col for col in attendance_df.columns if 'Attendance' in col]
    for col in attendance_columns:
        attendance_df[col] = attendance_df[col].astype(str)
        attendance_df[col] = attendance_df[col].apply(DataTypeValidator.validate_attendance)

    # Merge datasets
    merged_df = pd.merge(marks_df, attendance_df, on='Name', how='outer')

    # Handle missing values
    for col in marks_columns:
        merged_df[col] = merged_df[col].fillna(0)
    for col in attendance_columns:
        merged_df[col] = merged_df[col].fillna(0)

    return merged_df, marks_columns, attendance_columns

# =============================================================================
# Feature Engineering
# =============================================================================
def engineer_features(merged_df: pd.DataFrame, marks_columns: List[str], attendance_columns: List[str]) -> pd.DataFrame:
    """Generate additional features for analysis"""
    # Scale marks to 100-point scale
    for col, max_marks in MAX_MARKS_CONFIG.items():
        merged_df[f'{col}_Scaled'] = merged_df[col].apply(lambda x: scale_marks_to_100(x, max_marks))

    # Calculate total marks and percentage
    merged_df['Total Marks'] = merged_df[marks_columns].sum(axis=1)
    merged_df['Percentage Marks'] = (merged_df['Total Marks'] / sum(MAX_MARKS_CONFIG.values())) * 100

    # Calculate attendance percentage
    merged_df['Attendance Percentage'] = merged_df[attendance_columns].mean(axis=1) * 100

    # Calculate weighted percentage
    merged_df['Weighted Percentage'] = (
        (merged_df['Attendance Percentage'] * 0.4) +
        (merged_df['Mini Test 1_Scaled'] * 0.2) +
        (merged_df['Mini Test 2_Scaled'] * 0.2) +
        (merged_df['Live Test_Scaled'] * 0.4) +
        (merged_df['Assignment_Scaled'] * 0.2)
    )

    # Calculate consistency scores
    merged_df['Score Consistency'] = merged_df[marks_columns].std(axis=1)
    merged_df['Attendance Consistency'] = merged_df[attendance_columns].std(axis=1)

    # Calculate improvement trajectory
    merged_df['Test Improvement'] = merged_df['Mini Test 2'] - merged_df['Mini Test 1']

    # Classify performance
    merged_df['Performance Category'] = merged_df['Weighted Percentage'].apply(classify_performance)

    return merged_df

# =============================================================================
# Analysis Functions
# =============================================================================
def analyze_performance(merged_df: pd.DataFrame):
    """Perform detailed performance analysis"""
    print("\n=== Detailed Student Performance Analysis ===\n")

    # Basic Statistics
    print("Basic Statistics:")
    print(merged_df[['Mini Test 1', 'Mini Test 2', 'Live Test', 'Assignment', 'Weighted Percentage']].describe())

    # Attendance Impact Analysis
    print("\nAttendance Impact Analysis:")
    correlation = merged_df['Attendance Percentage'].corr(merged_df['Weighted Percentage'])
    print(f"Correlation between attendance and performance: {correlation:.2f}")

    # T-test for attendance groups
    high_attendance = merged_df[merged_df['Attendance Percentage'] >= 75]['Weighted Percentage']
    low_attendance = merged_df[merged_df['Attendance Percentage'] < 75]['Weighted Percentage']
    t_stat, p_value = stats.ttest_ind(high_attendance, low_attendance)
    print(f"T-test p-value for performance difference between high and low attendance groups: {p_value:.4f}")

    # Performance Categories
    print("\nPerformance Categories Distribution:")
    performance_dist = merged_df['Performance Category'].value_counts()
    print(performance_dist)
    print("\nPercentage Distribution:")
    print((performance_dist / len(merged_df) * 100).round(2))

    # Top Performers
    print("\nTop 3 Students Overall:")
    print(merged_df.nlargest(3, 'Weighted Percentage')[['Name', 'Weighted Percentage', 'Attendance Percentage']])

    # Students Needing Improvement
    print("\nStudents Needing Improvement (Weighted Percentage < 50%):")
    print(merged_df[merged_df['Weighted Percentage'] < 50][['Name', 'Weighted Percentage', 'Attendance Percentage']])

    # Attendance Concerns
    print("\nStudents with attendance below 75% but good performance (>50%):")
    attendance_concerns = merged_df[
        (merged_df['Attendance Percentage'] < 75) & 
        (merged_df['Weighted Percentage'] > 50)
    ]
    print(attendance_concerns[['Name', 'Attendance Percentage', 'Weighted Percentage']])

# =============================================================================
# Visualization Functions
# =============================================================================
def create_visualizations(merged_df: pd.DataFrame):
    """Generate all visualizations for the analysis"""
    print("\nGenerating visualizations...")
    
    # Set visualization style
    plt.style.use('default')  # Use default style as base
    sns.set_theme(style="whitegrid")  # Set seaborn style to whitegrid
    sns.set_palette(PLOT_STYLE['palette'])

    # Create main analysis plots
    create_main_analysis_plots(merged_df)
    
    # Create additional analysis plots
    create_additional_analysis_plots(merged_df)

def create_main_analysis_plots(merged_df: pd.DataFrame):
    """Create the main set of analysis plots"""
    # Create figure with white background
    fig = plt.figure(figsize=PLOT_STYLE['figsize'])
    fig.patch.set_facecolor('white')
    plt.gca().set_facecolor('white')  # Set axes background to white

    # 1. Top 5 students bar chart
    plt.subplot(2, 2, 1)
    top_5 = merged_df.nlargest(5, 'Weighted Percentage')
    if not top_5.empty:
        sns.barplot(data=top_5, x='Name', y='Weighted Percentage', 
                   ci='sd', capsize=0.05, errwidth=2)
        plt.xticks(rotation=45, ha='right')
        plt.title('Top 5 Students - Weighted Percentages')
        plt.ylabel('Weighted Percentage (%)')
        plt.gca().set_facecolor('white')  # Set subplot background to white

    # 2. Performance categories pie chart
    plt.subplot(2, 2, 2)
    performance_dist = merged_df['Performance Category'].value_counts()
    if not performance_dist.empty:
        colors = sns.color_palette("husl", n_colors=len(performance_dist))
        plt.pie(performance_dist, labels=performance_dist.index, autopct='%1.1f%%',
                colors=colors, explode=[0.05] * len(performance_dist))
        plt.title('Performance Categories Distribution')
        plt.gca().set_facecolor('white')  # Set subplot background to white

    # 3. Test scores distribution
    plt.subplot(2, 2, 3)
    test_data = merged_df[['Mini Test 1', 'Mini Test 2', 'Live Test', 'Assignment']]
    if not test_data.empty:
        test_data_melted = pd.melt(test_data)
        sns.violinplot(data=test_data_melted, x='variable', y='value', 
                      inner='box', cut=0)
        plt.title('Test Scores Distribution')
        plt.xticks(rotation=45)
        plt.xlabel('')
        plt.ylabel('Scores')
        plt.gca().set_facecolor('white')  # Set subplot background to white

    # 4. Attendance vs Performance correlation
    plt.subplot(2, 2, 4)
    if not merged_df.empty:
        sns.regplot(data=merged_df, 
                   x='Attendance Percentage', 
                   y='Weighted Percentage',
                   scatter_kws={'alpha':0.5},
                   line_kws={'color': 'red'},
                   ci=95)
        plt.title('Attendance vs Performance Correlation')
        plt.xlabel('Attendance Percentage (%)')
        plt.ylabel('Weighted Percentage (%)')
        plt.gca().set_facecolor('white')  # Set subplot background to white

    plt.tight_layout()
    
    # Save with white background
    plt.savefig('plots/main_analysis.png', 
                dpi=PLOT_STYLE['dpi'], 
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.close()

def create_additional_analysis_plots(merged_df: pd.DataFrame):
    """Create additional analysis plots"""
    # 1. Performance trends
    plt.figure(figsize=(12, 6))
    fig = plt.gcf()
    fig.patch.set_facecolor('white')
    plt.gca().set_facecolor('white')
    
    test_data = merged_df[['Mini Test 1', 'Mini Test 2', 'Live Test', 'Assignment']]
    if not test_data.empty:
        test_data_melted = pd.melt(test_data)
        sns.pointplot(data=test_data_melted, x='variable', y='value',
                     ci=95, capsize=0.2)
        plt.title('Performance Trends Across Tests')
        plt.xticks(rotation=45)
        plt.xlabel('')
        plt.ylabel('Average Score')
        plt.grid(True, alpha=0.3)
        plt.savefig('plots/performance_trends.png', 
                   dpi=PLOT_STYLE['dpi'], 
                   bbox_inches='tight',
                   facecolor='white',
                   edgecolor='none')
        plt.close()

    # 2. Correlation heatmap
    plt.figure(figsize=(10, 8))
    fig = plt.gcf()
    fig.patch.set_facecolor('white')
    plt.gca().set_facecolor('white')
    
    numerical_cols = ['Mini Test 1', 'Mini Test 2', 'Live Test', 'Assignment', 
                     'Attendance Percentage', 'Weighted Percentage']
    correlation_matrix = merged_df[numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                fmt='.2f', square=True)
    plt.title('Correlation Matrix of Performance Metrics')
    plt.tight_layout()
    plt.savefig('plots/correlation_matrix.png', 
                dpi=PLOT_STYLE['dpi'], 
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.close()

    # 3. Score distributions
    plt.figure(figsize=(15, 5))
    fig = plt.gcf()
    fig.patch.set_facecolor('white')
    
    for i, col in enumerate(['Mini Test 1', 'Mini Test 2', 'Live Test', 'Assignment']):
        plt.subplot(1, 4, i+1)
        sns.kdeplot(data=merged_df, x=col, fill=True)
        plt.title(f'{col} Distribution')
        plt.xlabel('Score')
        plt.gca().set_facecolor('white')
    plt.tight_layout()
    plt.savefig('plots/score_distributions.png', 
                dpi=PLOT_STYLE['dpi'], 
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.close()

    # 4. Box plots for each test
    create_box_plots(merged_df)

def create_box_plots(merged_df: pd.DataFrame):
    """Create box plots for each test to visualize score distribution and outliers"""
    # Create a figure with subplots for each test
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.patch.set_facecolor('white')
    fig.suptitle('Box Plots of Test Scores', fontsize=16, y=0.95)

    # List of tests to plot
    tests = ['Mini Test 1', 'Mini Test 2', 'Live Test', 'Assignment']
    
    # Create box plots for each test
    for idx, test in enumerate(tests):
        row = idx // 2
        col = idx % 2
        
        # Create box plot
        sns.boxplot(data=merged_df, y=test, ax=axes[row, col])
        
        # Add violin plot for additional distribution information
        sns.violinplot(data=merged_df, y=test, ax=axes[row, col], color='lightgray')
        
        # Customize the plot
        axes[row, col].set_title(f'{test} Score Distribution')
        axes[row, col].set_ylabel('Score')
        axes[row, col].set_facecolor('white')  # Set subplot background to white
        
        # Add statistical summary
        stats_text = f"Mean: {merged_df[test].mean():.1f}\n"
        stats_text += f"Median: {merged_df[test].median():.1f}\n"
        stats_text += f"Std: {merged_df[test].std():.1f}\n"
        stats_text += f"Outliers: {len(merged_df[merged_df[test] > merged_df[test].quantile(0.75) + 1.5 * (merged_df[test].quantile(0.75) - merged_df[test].quantile(0.25))])}"
        
        axes[row, col].text(0.95, 0.95, stats_text,
                          transform=axes[row, col].transAxes,
                          verticalalignment='top',
                          horizontalalignment='right',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig('plots/box_plots.png', 
                dpi=PLOT_STYLE['dpi'], 
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.close()

    # Create a combined box plot for all tests
    plt.figure(figsize=(12, 6))
    fig = plt.gcf()
    fig.patch.set_facecolor('white')
    plt.gca().set_facecolor('white')
    
    test_data = merged_df[['Mini Test 1', 'Mini Test 2', 'Live Test', 'Assignment']]
    test_data_melted = pd.melt(test_data)
    
    # Create box plot with violin plot overlay
    sns.boxplot(data=test_data_melted, x='variable', y='value')
    sns.violinplot(data=test_data_melted, x='variable', y='value', color='lightgray')
    
    plt.title('Combined Box Plots of All Tests')
    plt.xlabel('Test')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    
    # Add statistical summary
    stats_text = "Statistical Summary:\n"
    for test in tests:
        stats_text += f"\n{test}:\n"
        stats_text += f"Mean: {merged_df[test].mean():.1f}\n"
        stats_text += f"Median: {merged_df[test].median():.1f}\n"
        stats_text += f"Std: {merged_df[test].std():.1f}"
    
    plt.text(0.95, 0.95, stats_text,
             transform=plt.gca().transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('plots/combined_box_plots.png', 
                dpi=PLOT_STYLE['dpi'], 
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.close()

    # Print outlier information
    print("\nOutlier Analysis:")
    for test in tests:
        Q1 = merged_df[test].quantile(0.25)
        Q3 = merged_df[test].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = merged_df[(merged_df[test] < lower_bound) | (merged_df[test] > upper_bound)]
        print(f"\n{test}:")
        print(f"Number of outliers: {len(outliers)}")
        if len(outliers) > 0:
            print("Outlier students:")
            for _, row in outliers.iterrows():
                print(f"- {row['Name']}: {row[test]:.1f}")

# =============================================================================
# Results Export
# =============================================================================
def export_results(merged_df: pd.DataFrame, performance_dist: pd.Series, correlation_matrix: pd.DataFrame):
    """Export analysis results to Excel"""
    print("\nSaving results to Excel...")
    with pd.ExcelWriter('student_analysis_results.xlsx') as writer:
        merged_df.to_excel(writer, sheet_name='Complete Data', index=False)
        
        # Basic statistics
        stats_df = merged_df[['Mini Test 1', 'Mini Test 2', 'Live Test', 'Assignment', 
                             'Weighted Percentage']].describe()
        stats_df.to_excel(writer, sheet_name='Statistics')
        
        # Performance categories
        performance_summary = pd.DataFrame({
            'Category': performance_dist.index,
            'Count': performance_dist.values,
            'Percentage': (performance_dist / len(merged_df) * 100).round(2)
        })
        performance_summary.to_excel(writer, sheet_name='Performance Categories', index=False)
        
        # Attendance analysis
        attendance_analysis = merged_df[['Name', 'Attendance Percentage', 'Weighted Percentage']]\
            .sort_values('Attendance Percentage', ascending=False)
        attendance_analysis.to_excel(writer, sheet_name='Attendance Analysis', index=False)
        
        # Correlation matrix
        correlation_matrix.round(3).to_excel(writer, sheet_name='Correlations')

# =============================================================================
# Main Execution
# =============================================================================
def main():
    """Main execution function"""
    # Load data
    marks_df, attendance_df = load_data()
    
    # Preprocess data
    merged_df, marks_columns, attendance_columns = preprocess_data(marks_df, attendance_df)
    
    # Engineer features
    merged_df = engineer_features(merged_df, marks_columns, attendance_columns)
    
    # Analyze performance
    analyze_performance(merged_df)
    
    # Create visualizations
    create_visualizations(merged_df)
    
    # Export results
    performance_dist = merged_df['Performance Category'].value_counts()
    numerical_cols = ['Mini Test 1', 'Mini Test 2', 'Live Test', 'Assignment', 
                     'Attendance Percentage', 'Weighted Percentage']
    correlation_matrix = merged_df[numerical_cols].corr()
    export_results(merged_df, performance_dist, correlation_matrix)
    
    print("\nDetailed analysis has been saved to 'student_analysis_results.xlsx'")

if __name__ == "__main__":
    main() 