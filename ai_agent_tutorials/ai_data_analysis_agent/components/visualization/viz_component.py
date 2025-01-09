import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import time

def validate_visualization_data(df: pd.DataFrame, x_axis: str, y_axis: str) -> tuple[bool, str]:
    """Validate data for visualization"""
    if x_axis not in df.columns:
        return False, f"X-axis column '{x_axis}' not found in data"
    
    if y_axis not in df.columns:
        return False, f"Y-axis column '{y_axis}' not found in data"
    
    if df.empty:
        return False, "No data to visualize"
    
    if df[y_axis].isnull().all():
        return False, f"Y-axis column '{y_axis}' contains only null values"
        
    return True, ""

class VizComponent:
    def __init__(self, df: pd.DataFrame, viz_info: Dict[str, Any]):
        self.df = df.copy()
        self.viz_info = viz_info
        self.color_schemes = {
            'default': px.colors.qualitative.Set1,
            'sequential': px.colors.sequential.Blues,
            'categorical': px.colors.qualitative.Plotly,
            'diverging': px.colors.diverging.RdBu
        }
        
        # Handle NaN values
        self.df = self.df.fillna(0)

    def format_number(self, value: float) -> str:
        """Format numbers with K/M suffix"""
        if value >= 1e6:
            return f"{value/1e6:.1f}M"
        elif value >= 1e3:
            return f"{value/1e3:.1f}K"
        return f"{value:,.0f}"

    def create_bar_chart(self, x_axis: str, y_axis: str, chart_key: str) -> go.Figure:
        """Create an interactive bar chart"""
        fig = px.bar(
            self.df,
            x=x_axis,
            y=y_axis,
            title=f"{y_axis} by {x_axis}",
            color_discrete_sequence=self.color_schemes['default']
        )
        
        fig.update_layout(
            xaxis_title=x_axis,
            yaxis_title=y_axis,
            hovermode='x unified',
            height=500,
            template='plotly_dark'
        )
        
        if len(self.df[x_axis].unique()) > 10:
            fig.update_layout(xaxis_tickangle=-45)
        
        return fig

    def create_line_chart(self, x_axis: str, y_axis: str, chart_key: str) -> go.Figure:
        """Create an interactive line chart"""
        fig = px.line(
            self.df,
            x=x_axis,
            y=y_axis,
            title=f"Trend of {y_axis} over {x_axis}",
            markers=True
        )
        
        fig.update_layout(
            xaxis_title=x_axis,
            yaxis_title=y_axis,
            hovermode='x unified',
            height=500,
            template='plotly_dark'
        )
        
        return fig

    def create_scatter_chart(self, x_axis: str, y_axis: str, chart_key: str) -> go.Figure:
        """Create an interactive scatter plot"""
        fig = px.scatter(
            self.df,
            x=x_axis,
            y=y_axis,
            title=f"{y_axis} vs {x_axis}",
            trendline="ols" if len(self.df) > 1 else None
        )
        
        fig.update_layout(
            xaxis_title=x_axis,
            yaxis_title=y_axis,
            height=500,
            template='plotly_dark'
        )
        
        return fig

    def create_pie_chart(self, x_axis: str, y_axis: str, chart_key: str) -> go.Figure:
        """Create an interactive pie chart"""
        pie_data = self.df.groupby(x_axis)[y_axis].sum().reset_index()
        
        fig = px.pie(
            pie_data,
            values=y_axis,
            names=x_axis,
            title=f"Distribution of {y_axis} by {x_axis}",
            color_discrete_sequence=self.color_schemes['default']
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label'
        )
        
        fig.update_layout(height=500, template='plotly_dark')
        
        return fig

    def create_area_chart(self, x_axis: str, y_axis: str, chart_key: str) -> go.Figure:
        """Create an interactive area chart"""
        fig = px.area(
            self.df,
            x=x_axis,
            y=y_axis,
            title=f"Area plot of {y_axis} over {x_axis}"
        )
        
        fig.update_layout(
            xaxis_title=x_axis,
            yaxis_title=y_axis,
            hovermode='x unified',
            height=500,
            template='plotly_dark'
        )
        
        return fig

    def create_box_plot(self, x_axis: str, y_axis: str, chart_key: str) -> go.Figure:
        """Create an interactive box plot"""
        fig = px.box(
            self.df,
            x=x_axis,
            y=y_axis,
            title=f"Distribution of {y_axis} by {x_axis}"
        )
        
        fig.update_layout(
            xaxis_title=x_axis,
            yaxis_title=y_axis,
            height=500,
            template='plotly_dark'
        )
        
        return fig

    def render(self):
        """Render the visualization component"""
        try:
            # Create a container for the visualization
            viz_container = st.container()
            
            with viz_container:
                # Create columns for controls
                col1, col2, col3 = st.columns(3)
                
                # Chart type selector
                with col1:
                    chart_type = st.selectbox(
                        "Chart Type",
                        options=['bar', 'line', 'scatter', 'pie', 'area', 'box'],
                        index=['bar', 'line', 'scatter', 'pie', 'area', 'box'].index(
                            self.viz_info.get('recommended_viz', 'bar')
                        )
                    )

                # X-axis selector
                with col2:
                    x_axis = st.selectbox(
                        "X Axis",
                        options=self.viz_info.get('columns', []),
                        index=self.viz_info.get('columns', []).index(
                            self.viz_info.get('x_axis', self.viz_info.get('columns', [])[0])
                        )
                    )

                # Y-axis selector
                with col3:
                    y_axis = st.selectbox(
                        "Y Axis",
                        options=self.viz_info.get('numerical_columns', []),
                        index=self.viz_info.get('numerical_columns', []).index(
                            self.viz_info.get('y_axis', self.viz_info.get('numerical_columns', [])[0])
                        )
                    )

                # Validate the data
                is_valid, error_message = validate_visualization_data(self.df, x_axis, y_axis)
                if not is_valid:
                    st.error(error_message)
                    return

                # Generate a unique key for this chart
                chart_key = f"viz_{hash(f'{chart_type}_{x_axis}_{y_axis}_{time.time()}')}"

                # Create and display the appropriate chart
                chart_creators = {
                    'bar': self.create_bar_chart,
                    'line': self.create_line_chart,
                    'scatter': self.create_scatter_chart,
                    'pie': self.create_pie_chart,
                    'area': self.create_area_chart,
                    'box': self.create_box_plot
                }

                if chart_type in chart_creators:
                    fig = chart_creators[chart_type](x_axis, y_axis, chart_key)
                    st.plotly_chart(fig, use_container_width=True, key=chart_key)
                else:
                    st.error(f"Unsupported chart type: {chart_type}")
                    return
                
                # Display visualization reason if available
                if 'reason' in self.viz_info:
                    st.info(f"📊 {self.viz_info['reason']}")

                # Display chart type descriptions
                with st.expander("📈 Chart Type Descriptions"):
                    st.markdown("""
                    ### Available Chart Types:
                    
                    - **Bar Chart**: Shows comparisons between categories. Best for comparing quantities across different groups.
                    - **Line Chart**: Shows trends over time or sequences. Ideal for showing how data changes over a continuous interval.
                    - **Scatter Plot**: Shows relationships between two variables. Great for identifying correlations and patterns.
                    - **Pie Chart**: Shows parts of a whole. Best when you want to show proportions and percentages.
                    - **Area Chart**: Similar to line chart but with filled areas. Good for showing cumulative totals over time.
                    - **Box Plot**: Shows statistical distribution of data. Excellent for showing data spread, quartiles, and outliers.
                    
                    Choose the chart type that best fits your data and analysis goals!
                    """)

        except Exception as e:
            st.error(f"Error in visualization component: {str(e)}")
            st.error("Please check your data and visualization settings.")

def create_visualization(df: pd.DataFrame, viz_info: Dict[str, Any]) -> None:
    """Helper function to create and display visualization"""
    viz = VizComponent(df, viz_info)
    viz.render()
