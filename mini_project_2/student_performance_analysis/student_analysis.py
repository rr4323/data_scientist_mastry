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

class DataTypeValidator:
    """Class to handle data type validation and conversion"""
    
    @staticmethod
    def validate_numeric(value: Any) -> Union[float, None]:
        """Validate and convert numeric values"""
        if pd.isna(value):
            return None
        try:
            # Remove any currency symbols, commas, or other non-numeric characters
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
            # Log unexpected values
            print(f"Warning: Unexpected attendance value '{value}' found. Defaulting to 0.")
            return 0

def validate_and_clean_data(df: pd.DataFrame, column_types: Dict[str, str]) -> pd.DataFrame:
    """
    Validate and clean DataFrame based on specified column types
    
    Args:
        df: Input DataFrame
        column_types: Dictionary mapping column names to their expected types
                     ('numeric', 'string', 'attendance')
    """
    validator = DataTypeValidator()
    
    for column, dtype in column_types.items():
        if column not in df.columns:
            print(f"Warning: Column '{column}' not found in DataFrame")
            continue
            
        if dtype == 'numeric':
            df[column] = df[column].apply(validator.validate_numeric)
            # Convert any remaining None values to NaN for pandas operations
            df[column] = pd.to_numeric(df[column], errors='coerce')
            
        elif dtype == 'string':
            df[column] = df[column].apply(validator.validate_string)
            
        elif dtype == 'attendance':
            df[column] = df[column].apply(validator.validate_attendance)
            
    return df

def validate_marks(df, columns):
    """Validate marks are within acceptable range"""
    for col in columns:
        # First ensure column is numeric
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
        mask = (df[col] < 0) | (df[col] > 100)
        if mask.any():
            print(f"Warning: Found invalid marks in {col}")
            # Replace invalid marks with median of valid marks
            valid_median = df[~mask][col].median()
            df.loc[mask, col] = valid_median
    return df

def detect_outliers(df, columns, threshold=3):
    """Detect and handle outliers using Z-score method"""
    for col in columns:
        # Ensure column is numeric before calculating z-scores
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Warning: Column {col} is not numeric. Converting to numeric...")
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        z_scores = np.abs(stats.zscore(df[col]))
        outliers = z_scores > threshold
        if outliers.any():
            print(f"Found {outliers.sum()} outliers in {col}")
            # Replace outliers with column median
            df.loc[outliers, col] = df[col].median()
    return df

def normalize_names(name):
    """Advanced name normalization"""
    if pd.isna(name):
        return ""
    # Remove extra spaces
    name = ' '.join(str(name).split())
    # Convert to title case
    name = name.title()
    # Remove special characters but keep valid name characters
    name = re.sub(r'[^a-zA-Z\s\'-]', '', name)
    # Handle multiple spaces, hyphens
    name = re.sub(r'\s+', ' ', name)
    name = re.sub(r'-+', '-', name)
    return name.strip()

# Read data from Excel file
try:
    marks_df = pd.read_excel('student.xlsx', sheet_name='Marks')
    attendance_df = pd.read_excel('student.xlsx', sheet_name='Attendance')
except FileNotFoundError:
    print("Error: 'student.xlsx' file not found!")
    exit(1)
except Exception as e:
    print(f"Error reading Excel file: {e}")
    exit(1)

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
# Add attendance columns dynamically
attendance_column_types.update({
    col: 'attendance' for col in attendance_df.columns 
    if col.startswith('Attendance') or col != 'Name'
})

# Validate and clean data types
print("\nValidating and cleaning data types...")
marks_df = validate_and_clean_data(marks_df, marks_column_types)
attendance_df = validate_and_clean_data(attendance_df, attendance_column_types)

# 1. Advanced Name Cleaning
print("\nCleaning and normalizing student names...")
marks_df['Name'] = marks_df['Name'].apply(normalize_names)
attendance_df['Name'] = attendance_df['Name'].apply(normalize_names)

# 2. Data Validation for Marks
print("\nValidating marks data...")
marks_columns = ['Mini Test 1', 'Mini Test 2', 'Live Test', 'Assignment']
columns = marks_df.columns
marks_df = validate_marks(marks_df, marks_columns)

# 3. Outlier Detection and Handling for Marks
print("\nChecking for outliers in marks...")
marks_df = detect_outliers(marks_df, marks_columns)

# 4. Advanced Attendance Processing
print("\nProcessing attendance data...")
attendance_columns = [col for col in attendance_df.columns if 'Attendance' in col]
for col in attendance_columns:
    # Convert column to string type first
    attendance_df[col] = attendance_df[col].astype(str)
    
    # Now process the attendance values
    def process_attendance(value):
        if pd.isna(value):
            return 0
        value = str(value).strip().upper()
        positive_markers = {'Y', 'YES', 'PRESENT', '1', 'P', 'TRUE', '1.0'}
        negative_markers = {'N', 'NO', 'ABSENT', '0', 'A', 'FALSE', '0.0'}
        
        if value in positive_markers:
            return 1
        elif value in negative_markers:
            return 0
        else:
            print(f"Warning: Unexpected attendance value '{value}' found. Defaulting to 0.")
            return 0
    
    attendance_df[col] = attendance_df[col].apply(process_attendance)

# 5. Merge DataFrames with Validation
print("\nMerging datasets with validation...")
merged_df = pd.merge(marks_df, attendance_df, on='Name', how='outer')

# 6. Advanced Missing Value Handling
print("\nHandling missing values...")
# For marks: Fill missing values with 0
print("Filling missing marks with 0...")
for col in marks_columns:
    if merged_df[col].isnull().any():
        missing_count = merged_df[col].isnull().sum()
        print(f"Found {missing_count} missing values in {col}")
        merged_df[col] = merged_df[col].fillna(0)

# For attendance: Keep existing handling (0 for missing)
for col in attendance_columns:
    if merged_df[col].isnull().any():
        missing_count = merged_df[col].isnull().sum()
        print(f"Found {missing_count} missing attendance records in {col}")
        merged_df[col] = merged_df[col].fillna(0)

# 7. Feature Engineering
print("\nGenerating additional features...")
# Calculate consistency scores
merged_df['Score Consistency'] = merged_df[marks_columns].std(axis=1)
merged_df['Attendance Consistency'] = merged_df[attendance_columns].std(axis=1)

# Calculate improvement trajectory
merged_df['Test Improvement'] = merged_df['Mini Test 2'] - merged_df['Mini Test 1']

# 8. Data Standardization
print("\nStandardizing numerical features...")
# Create a copy of the original marks for later use
original_marks = merged_df[marks_columns].copy()

# Standardize only the marks columns
scaler = StandardScaler()
merged_df[marks_columns] = scaler.fit_transform(merged_df[marks_columns])

# 9. Data Quality Report
print("\nGenerating data quality report...")
print(f"Total number of students: {len(merged_df)}")
print(f"Number of complete records: {merged_df.dropna().shape[0]}")
print("\nColumn-wise completeness:")
print((merged_df.count() / len(merged_df) * 100).round(2))

# Reset scaled values back to original scale for analysis
merged_df[marks_columns] = original_marks

print("\n=== Preprocessing Complete ===\n")

# Calculate total and percentage marks
merged_df['Total Marks'] = merged_df[['Mini Test 1', 'Mini Test 2', 'Live Test', 'Assignment']].sum(axis=1)
max_marks = 200  # Assuming max marks: Mini Tests(20) + Live Test(120) + Assignment(60)
merged_df['Percentage Marks'] = (merged_df['Total Marks'] / max_marks) * 100

# Calculate attendance percentage
merged_df['Attendance Percentage'] = merged_df[attendance_columns].mean(axis=1) * 100

# Calculate weighted percentage
merged_df['Weighted Percentage'] = (
    (merged_df['Attendance Percentage'] * 0.4) +
    (merged_df['Mini Test 1'] * 0.1) +
    (merged_df['Mini Test 2'] * 0.1) +
    (merged_df['Live Test'] * 0.2) +
    (merged_df['Assignment'] * 0.2)
)

# Classify performance
def classify_performance(percentage):
    if percentage >= 85:
        return 'Excellent'
    elif 71 <= percentage < 85:
        return 'Good'
    elif 50 <= percentage < 71:
        return 'Average'
    else:
        return 'Needs Improvement'

merged_df['Performance Category'] = merged_df['Weighted Percentage'].apply(classify_performance)

# Enhanced Analysis Section
print("\n=== Detailed Student Performance Analysis ===\n")

# 1. Basic Statistics
print("Basic Statistics:")
print(merged_df[['Mini Test 1', 'Mini Test 2', 'Live Test', 'Assignment', 'Weighted Percentage']].describe())

# 2. Attendance Impact Analysis
print("\nAttendance Impact Analysis:")
correlation = merged_df['Attendance Percentage'].corr(merged_df['Weighted Percentage'])
print(f"Correlation between attendance and performance: {correlation:.2f}")

# Perform t-test to compare performance between high and low attendance groups
high_attendance = merged_df[merged_df['Attendance Percentage'] >= 75]['Weighted Percentage']
low_attendance = merged_df[merged_df['Attendance Percentage'] < 75]['Weighted Percentage']
t_stat, p_value = stats.ttest_ind(high_attendance, low_attendance)
print(f"T-test p-value for performance difference between high and low attendance groups: {p_value:.4f}")

# 3. Performance Categories Analysis
print("\nPerformance Categories Distribution:")
performance_dist = merged_df['Performance Category'].value_counts()
print(performance_dist)
print("\nPercentage Distribution:")
print((performance_dist / len(merged_df) * 100).round(2))

# 4. Top Performers Analysis
print("\nTop 3 Students Overall:")
print(merged_df.nlargest(3, 'Weighted Percentage')[['Name', 'Weighted Percentage', 'Attendance Percentage']])

# 5. Improvement Needed
print("\nStudents Needing Improvement (Weighted Percentage < 50%):")
print(merged_df[merged_df['Weighted Percentage'] < 50][['Name', 'Weighted Percentage', 'Attendance Percentage']])

# 6. Attendance Concerns
print("\nStudents with attendance below 75% but good performance (>50%):")
attendance_concerns = merged_df[
    (merged_df['Attendance Percentage'] < 75) & 
    (merged_df['Weighted Percentage'] > 50)
]
print(attendance_concerns[['Name', 'Attendance Percentage', 'Weighted Percentage']])

# Enhanced Visualizations
print("\nGenerating visualizations...")

# Set seaborn style and color palette
sns.set_style("whitegrid")
sns.set_palette("husl")

# Create figure with subplots
fig = plt.figure(figsize=(15, 10))

# 1. Bar chart for top 5 students with error bars
plt.subplot(2, 2, 1)
top_5 = merged_df.nlargest(5, 'Weighted Percentage')
if not top_5.empty:
    sns.barplot(data=top_5, x='Name', y='Weighted Percentage', 
                ci='sd', capsize=0.05, errwidth=2)
    plt.xticks(rotation=45, ha='right')
    plt.title('Top 5 Students - Weighted Percentages')
    plt.ylabel('Weighted Percentage (%)')
else:
    plt.text(0.5, 0.5, 'No data available', ha='center', va='center')

# 2. Enhanced pie chart for performance categories
plt.subplot(2, 2, 2)
if not performance_dist.empty:
    colors = sns.color_palette("husl", n_colors=len(performance_dist))
    plt.pie(performance_dist, labels=performance_dist.index, autopct='%1.1f%%',
            colors=colors, 
            explode=[0.05] * len(performance_dist))
    plt.title('Performance Categories Distribution')
else:
    plt.text(0.5, 0.5, 'No data available', ha='center', va='center')

# 3. Enhanced box plots with violin plots
plt.subplot(2, 2, 3)
test_data = merged_df[['Mini Test 1', 'Mini Test 2', 'Live Test', 'Assignment']]
if not test_data.empty:
    # Melt the dataframe for seaborn
    test_data_melted = pd.melt(test_data)
    sns.violinplot(data=test_data_melted, x='variable', y='value', 
                  inner='box', cut=0)
    plt.title('Test Scores Distribution')
    plt.xticks(rotation=45)
    plt.xlabel('')
    plt.ylabel('Scores')
else:
    plt.text(0.5, 0.5, 'No data available', ha='center', va='center')

# 4. Scatter plot with regression line and confidence interval
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
else:
    plt.text(0.5, 0.5, 'No data available', ha='center', va='center')

plt.tight_layout()
plt.show()
plt.savefig('plots/main_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Additional visualizations using seaborn

# 1. Performance trends with confidence intervals
plt.figure(figsize=(12, 6))
if not test_data.empty:
    sns.pointplot(data=test_data_melted, x='variable', y='value',
                 ci=95, capsize=0.2)
    plt.title('Performance Trends Across Tests')
    plt.xticks(rotation=45)
    plt.xlabel('')
    plt.ylabel('Average Score')
    plt.grid(True, alpha=0.3)
else:
    plt.text(0.5, 0.5, 'No data available', ha='center', va='center')

plt.tight_layout()
plt.show()
plt.savefig('plots/performance_trends.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Correlation heatmap of numerical variables
plt.figure(figsize=(10, 8))
numerical_cols = ['Mini Test 1', 'Mini Test 2', 'Live Test', 'Assignment', 
                 'Attendance Percentage', 'Weighted Percentage']
correlation_matrix = merged_df[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            fmt='.2f', square=True)
plt.title('Correlation Matrix of Performance Metrics')
plt.tight_layout()
plt.show()
plt.savefig('plots/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Distribution of scores with kernel density estimation
plt.figure(figsize=(15, 5))
for i, col in enumerate(['Mini Test 1', 'Mini Test 2', 'Live Test', 'Assignment']):
    plt.subplot(1, 4, i+1)
    sns.kdeplot(data=merged_df, x=col, fill=True)
    plt.title(f'{col} Distribution')
    plt.xlabel('Score')
plt.tight_layout()
plt.show()
plt.savefig('plots/score_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

# Save detailed results to Excel with multiple sheets
print("\nSaving results to Excel...")
with pd.ExcelWriter('student_analysis_results.xlsx') as writer:
    merged_df.to_excel(writer, sheet_name='Complete Data', index=False)
    
    # Basic statistics sheet
    stats_df = merged_df[['Mini Test 1', 'Mini Test 2', 'Live Test', 'Assignment', 
                         'Weighted Percentage']].describe()
    stats_df.to_excel(writer, sheet_name='Statistics')
    
    # Performance categories sheet
    performance_summary = pd.DataFrame({
        'Category': performance_dist.index,
        'Count': performance_dist.values,
        'Percentage': (performance_dist / len(merged_df) * 100).round(2)
    })
    performance_summary.to_excel(writer, sheet_name='Performance Categories', index=False)
    
    # Attendance analysis sheet
    attendance_analysis = merged_df[['Name', 'Attendance Percentage', 'Weighted Percentage']]\
        .sort_values('Attendance Percentage', ascending=False)
    attendance_analysis.to_excel(writer, sheet_name='Attendance Analysis', index=False)
    
    # Correlation matrix sheet
    correlation_matrix.round(3).to_excel(writer, sheet_name='Correlations')

print("\nDetailed analysis has been saved to 'student_analysis_results.xlsx'") 