import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List
import pandas as pd
import numpy as np

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
            height=500,
            template='plotly_dark'
        )
        
        if len(self.df[x_axis].unique()) > 10:
            fig.update_layout(xaxis_tickangle=-45)
        
        return fig

    def create_pie_chart(self, x_axis: str, y_axis: str) -> go.Figure:
        """Create an interactive pie chart"""
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
        
        fig.update_layout(height=500, template='plotly_dark')
        
        return fig

    def render(self):
        """Render the visualization component"""
        try:
            # Generate a unique identifier for this instance
            viz_id = str(hash(frozenset(self.df.columns)))[:8]
            
            # Initialize session state keys for this visualization
            chart_type_key = f'chart_type_{viz_id}'
            x_axis_key = f'x_axis_{viz_id}'
            y_axis_key = f'y_axis_{viz_id}'
            
            # Initialize session state if not exists
            if chart_type_key not in st.session_state:
                st.session_state[chart_type_key] = self.viz_info.get('recommended_viz', 'bar')
            if x_axis_key not in st.session_state:
                st.session_state[x_axis_key] = self.viz_info.get('x_axis')
            if y_axis_key not in st.session_state:
                st.session_state[y_axis_key] = self.viz_info.get('y_axis')
            
            # Create columns for controls
            col1, col2, col3 = st.columns(3)
            
            # Chart type selector
            with col1:
                chart_type = st.selectbox(
                    "Chart Type",
                    options=self.viz_info.get('possible_viz', ['bar']),
                    index=self.viz_info.get('possible_viz', ['bar']).index(
                        st.session_state[chart_type_key]
                    ),
                    key=f'viz_chart_type_{viz_id}'
                )
                st.session_state[chart_type_key] = chart_type

            # X-axis selector
            with col2:
                x_axis = st.selectbox(
                    "X Axis",
                    options=self.viz_info.get('columns', []),
                    index=self.viz_info.get('columns', []).index(
                        st.session_state[x_axis_key]
                    ) if st.session_state[x_axis_key] in self.viz_info.get('columns', []) else 0,
                    key=f'viz_x_axis_{viz_id}'
                )
                st.session_state[x_axis_key] = x_axis

            # Y-axis selector
            with col3:
                y_axis = st.selectbox(
                    "Y Axis",
                    options=self.viz_info.get('numerical_columns', []),
                    index=self.viz_info.get('numerical_columns', []).index(
                        st.session_state[y_axis_key]
                    ) if st.session_state[y_axis_key] in self.viz_info.get('numerical_columns', []) else 0,
                    key=f'viz_y_axis_{viz_id}'
                )
                st.session_state[y_axis_key] = y_axis

            # Validate the data
            is_valid, error_message = validate_visualization_data(self.df, x_axis, y_axis)
            if not is_valid:
                st.error(error_message)
                return

            # Create and display the appropriate chart
            chart_key = f'chart_{viz_id}'
            if chart_type == 'bar':
                fig = self.create_bar_chart(x_axis, y_axis)
            elif chart_type == 'pie':
                fig = self.create_pie_chart(x_axis, y_axis)
            else:
                st.error(f"Unsupported chart type: {chart_type}")
                return

            # Display the chart
            st.plotly_chart(fig, use_container_width=True, key=chart_key)
            
            # Display visualization reason if available
            if 'reason' in self.viz_info:
                st.info(f"📊 {self.viz_info['reason']}", key=f'viz_info_{viz_id}')

        except Exception as e:
            st.error(f"Error in visualization component: {str(e)}")
            st.error("Please check your data and visualization settings.")

def create_visualization(df: pd.DataFrame, viz_info: Dict[str, Any]) -> None:
    """Helper function to create and display visualization"""
    viz = VizComponent(df, viz_info)
    viz.render()