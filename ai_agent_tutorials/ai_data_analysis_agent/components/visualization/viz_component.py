import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List
import pandas as pd
import numpy as np

def validate_visualization_data(df: pd.DataFrame, x_axis: str, y_axis: str) -> tuple[bool, str]:
    """
    Validate data for visualization
    Returns: (is_valid, error_message)
    """
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
        self.df = df.copy()  # Make a copy to prevent modifications
        self.viz_info = viz_info
        self.color_schemes = {
            'default': px.colors.qualitative.Set1,
            'sequential': px.colors.sequential.Blues,
            'categorical': px.colors.qualitative.Plotly,
            'diverging': px.colors.diverging.RdBu
        }
        
        # Handle NaN values
        self.df = self.df.fillna(0)  # or another appropriate strategy
        
    def format_number(self, value: float) -> str:
        """Format numbers with K/M suffix"""
        if value >= 1e6:
            return f"{value/1e6:.1f}M"
        elif value >= 1e3:
            return f"{value/1e3:.1f}K"
        return f"{value:,.0f}"

    def create_safe_figure(self, chart_func, x_axis: str, y_axis: str) -> go.Figure:
        """Safely create a figure with error handling"""
        try:
            # Validate data
            is_valid, error_message = validate_visualization_data(self.df, x_axis, y_axis)
            if not is_valid:
                fig = go.Figure()
                fig.add_annotation(
                    text=error_message,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False
                )
                return fig
            
            # Try to create the chart
            return chart_func(x_axis, y_axis)
            
        except Exception as e:
            # Create error figure
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating visualization: {str(e)}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False
            )
            return fig

    def create_bar_chart(self, x_axis: str, y_axis: str) -> go.Figure:
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
            height=500
        )
        
        if len(self.df[x_axis].unique()) > 10:
            fig.update_layout(xaxis_tickangle=-45)
        
        return fig

    def create_line_chart(self, x_axis: str, y_axis: str) -> go.Figure:
        """Create an interactive line chart"""
        fig = px.line(
            self.df,
            x=x_axis,
            y=y_axis,
            title=f"{y_axis} over {x_axis}",
            markers=True
        )
        
        fig.update_traces(
            line=dict(color=self.color_schemes['default'][0]),
            marker=dict(size=6)
        )
        
        fig.update_layout(
            xaxis_title=x_axis,
            yaxis_title=y_axis,
            hovermode='x unified',
            height=500
        )
        
        return fig

    def create_scatter_plot(self, x_axis: str, y_axis: str) -> go.Figure:
        """Create an interactive scatter plot"""
        fig = px.scatter(
            self.df,
            x=x_axis,
            y=y_axis,
            title=f"{y_axis} vs {x_axis}",
            trendline="ols" if len(self.df) > 2 else None
        )
        
        fig.update_traces(
            marker=dict(
                size=8,
                color=self.color_schemes['default'][0],
                line=dict(width=1, color='DarkSlateGrey')
            )
        )
        
        fig.update_layout(
            xaxis_title=x_axis,
            yaxis_title=y_axis,
            height=500
        )
        
        return fig

    def create_pie_chart(self, x_axis: str, y_axis: str) -> go.Figure:
        """Create an interactive pie chart"""
        # Aggregate data for pie chart
        pie_data = self.df.groupby(x_axis)[y_axis].sum().reset_index()
        
        fig = px.pie(
            pie_data,
            values=y_axis,
            names=x_axis,
            title=f"{y_axis} Distribution by {x_axis}",
            color_discrete_sequence=self.color_schemes['default']
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label'
        )
        
        fig.update_layout(height=500)
        
        return fig

    def render(self):
        """Render the visualization component in Streamlit with error handling"""
        try:
            # Create columns for controls
            col1, col2, col3 = st.columns(3)
            
            # Chart type selector
            with col1:
                chart_type = st.selectbox(
                    "Chart Type",
                    options=self.viz_info.get('possible_viz', ['bar']),
                    index=self.viz_info.get('possible_viz', ['bar']).index(
                        self.viz_info.get('recommended_viz', 'bar')
                    ),
                    help="Select the type of visualization"
                )

            # X-axis selector
            with col2:
                x_axis = st.selectbox(
                    "X Axis",
                    options=self.viz_info.get('columns', []),
                    index=self.viz_info.get('columns', []).index(
                        self.viz_info.get('x_axis', self.viz_info.get('columns', [])[0])
                    ) if self.viz_info.get('columns') else 0,
                    help="Select the column for X axis"
                )

            # Y-axis selector
            with col3:
                y_axis = st.selectbox(
                    "Y Axis",
                    options=self.viz_info.get('numerical_columns', []),
                    index=self.viz_info.get('numerical_columns', []).index(
                        self.viz_info.get('y_axis', self.viz_info.get('numerical_columns', [])[0])
                    ) if self.viz_info.get('numerical_columns') else 0,
                    help="Select the column for Y axis"
                )

            # Create and display the appropriate chart
            chart_functions = {
                'bar': self.create_bar_chart,
                'line': self.create_line_chart,
                'scatter': self.create_scatter_plot,
                'pie': self.create_pie_chart
            }
            
            if chart_type in chart_functions:
                fig = self.create_safe_figure(chart_functions[chart_type], x_axis, y_axis)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display visualization reason if available
                if 'reason' in self.viz_info:
                    st.info(f"📊 {self.viz_info['reason']}")
            else:
                st.error(f"Unsupported chart type: {chart_type}")

        except Exception as e:
            st.error(f"Error in visualization component: {str(e)}")
            st.error("Please check your data and visualization settings.")

def create_visualization(df: pd.DataFrame, viz_info: Dict[str, Any]) -> None:
    """Helper function to create and display visualization"""
    viz = VizComponent(df, viz_info)
    viz.render()