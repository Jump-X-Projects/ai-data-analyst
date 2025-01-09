import pandas as pd
import numpy as np
from typing import Dict, List, Any

def analyze_data_for_visualization(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyzes DataFrame and suggests appropriate visualizations"""
    # Get basic data characteristics
    columns = df.columns.tolist()
    
    # Identify column types
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Check for datetime-like string columns
    for col in categorical_cols:
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                pd.to_datetime(df[col])
                datetime_cols.append(col)
                categorical_cols.remove(col)
            except:
                pass

    result = {
        'possible_viz': [],
        'x_axis': columns[0],
        'y_axis': numerical_cols[0] if numerical_cols else columns[0],
        'columns': columns,
        'numerical_columns': numerical_cols,
        'categorical_columns': categorical_cols,
        'datetime_columns': datetime_cols
    }

    # Determine best visualization based on data characteristics
    if datetime_cols and numerical_cols:
        result.update({
            'recommended_viz': 'line',
            'possible_viz': ['line', 'bar'],
            'x_axis': datetime_cols[0],
            'y_axis': numerical_cols[0],
            'reason': 'Time series data detected - showing trends over time'
        })
    elif categorical_cols and numerical_cols:
        unique_cats = df[categorical_cols[0]].nunique()
        if unique_cats <= 10:
            result.update({
                'recommended_viz': 'bar',
                'possible_viz': ['bar', 'pie'],
                'x_axis': categorical_cols[0],
                'y_axis': numerical_cols[0],
                'reason': 'Categorical data with numerical values - comparing across categories'
            })
        else:
            result.update({
                'recommended_viz': 'bar',
                'possible_viz': ['bar'],
                'x_axis': categorical_cols[0],
                'y_axis': numerical_cols[0],
                'reason': 'Multiple categories with numerical values - showing distribution'
            })
    elif len(numerical_cols) >= 2:
        result.update({
            'recommended_viz': 'scatter',
            'possible_viz': ['scatter', 'line'],
            'x_axis': numerical_cols[0],
            'y_axis': numerical_cols[1],
            'reason': 'Multiple numerical columns - exploring relationships'
        })
    else:
        result.update({
            'recommended_viz': 'bar',
            'possible_viz': ['bar', 'line'],
            'reason': 'General data exploration'
        })

    return result

def prepare_data_for_visualization(df: pd.DataFrame) -> List[Dict]:
    """Prepares DataFrame for visualization"""
    df = df.copy()
    
    # Convert datetime columns to string format
    for col in df.select_dtypes(include=['datetime64']):
        df[col] = df[col].dt.strftime('%Y-%m-%d')
    
    # Round numerical values to 2 decimal places
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df[numerical_cols] = df[numerical_cols].round(2)
    
    # Handle any NaN values
    df = df.fillna('N/A')
    
    # Convert to list of dictionaries
    return df.to_dict('records')

def format_data_for_visualization(df: pd.DataFrame, x_axis: str, y_axis: str) -> pd.DataFrame:
    """Formats data specifically for visualization"""
    df = df.copy()
    
    # Sort by x-axis if it's a datetime
    try:
        if pd.api.types.is_datetime64_any_dtype(df[x_axis]):
            df = df.sort_values(x_axis)
    except:
        pass
    
    # Aggregate data if needed (e.g., for categorical x-axis)
    if df[x_axis].dtype == 'object' and pd.api.types.is_numeric_dtype(df[y_axis]):
        df = df.groupby(x_axis)[y_axis].agg(['sum', 'mean']).reset_index()
        df[y_axis] = df['sum']  # Default to sum for aggregation
    
    return df

def handle_large_datasets(df: pd.DataFrame, max_points: int = 1000) -> pd.DataFrame:
    """Handles large datasets by sampling or aggregating data"""
    if len(df) > max_points:
        # If time series, resample to reduce points
        if any('date' in col.lower() or 'time' in col.lower() for col in df.columns):
            date_col = next(col for col in df.columns if 'date' in col.lower() or 'time' in col.lower())
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
            
            # Calculate appropriate frequency
            total_duration = (df.index.max() - df.index.min()).total_seconds()
            freq = pd.Timedelta(seconds=int(total_duration / max_points))
            
            # Resample and aggregate
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df = df[numeric_cols].resample(freq).mean().reset_index()
        else:
            # If not time series, use systematic sampling
            df = df.iloc[::len(df)//max_points].reset_index(drop=True)
    
    return df

def enhance_visualization(viz_info: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    """Enhances visualization settings based on data analysis"""
    # Add formatting suggestions
    viz_info['formatting'] = {
        'number_format': {},
        'date_format': '%Y-%m-%d',
        'color_scheme': 'default'
    }
    
    # Analyze numerical columns for formatting
    for col in viz_info['numerical_columns']:
        max_val = df[col].max()
        if max_val > 1000000:
            viz_info['formatting']['number_format'][col] = 'M'  # Millions
        elif max_val > 1000:
            viz_info['formatting']['number_format'][col] = 'K'  # Thousands
        else:
            viz_info['formatting']['number_format'][col] = ','  # Regular comma formatting
    
    # Detect if color scheme should be different (e.g., for sequential data)
    if any(col for col in df.columns if 'percentage' in col.lower() or 'ratio' in col.lower()):
        viz_info['formatting']['color_scheme'] = 'sequential'
    
    return viz_info