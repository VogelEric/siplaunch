#!/usr/bin/env python3
"""
Event Prediction Visualization Tool
Generates a static HTML webpage with interactive charts for event prediction data.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
from datetime import datetime
from jinja2 import Template
import os
from pathlib import Path
import numpy as np


class PredictionDataProcessor:
    """Handles loading, processing, and analyzing prediction data."""

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data = None
        self.processed_data = {}

    def load_data(self):
        """Load and validate CSV data."""
        try:
            self.data = pd.read_csv(self.csv_path)
            # Parse dates with proper format
            self.data['Date of Prediction'] = pd.to_datetime(self.data['Date of Prediction'], format='%m/%d/%Y')
            self.data['Predicted Event Date'] = pd.to_datetime(self.data['Predicted Event Date'], format='%m/%d/%Y')

            # Calculate prediction windows (days between prediction and predicted event)
            self.data['Prediction Window'] = (self.data['Predicted Event Date'] - self.data['Date of Prediction']).dt.days

            print(f"Loaded {len(self.data)} prediction records")
            print(f"Events found: {self.data['Event'].unique().tolist()}")

        except Exception as e:
            raise ValueError(f"Error loading data: {e}")

    def get_unique_events(self):
        """Get list of all unique events for filtering."""
        return sorted(self.data['Event'].unique().tolist())

    def filter_by_event(self, event_name):
        """Filter data for a specific event."""
        return self.data[self.data['Event'] == event_name].copy()

    def calculate_event_statistics(self, event_name=None):
        """Calculate statistical measures for events."""
        if event_name:
            filtered_data = self.filter_by_event(event_name)
        else:
            filtered_data = self.data

        stats = {}
        for event in filtered_data['Event'].unique():
            event_data = filtered_data[filtered_data['Event'] == event]

            stats[event] = {
                'count': len(event_data),
                'avg_prediction_window': event_data['Prediction Window'].mean(),
                'median_prediction_window': event_data['Prediction Window'].median(),
                'std_prediction_window': event_data['Prediction Window'].std(),
                'min_prediction_window': event_data['Prediction Window'].min(),
                'max_prediction_window': event_data['Prediction Window'].max(),
                'earliest_prediction': event_data['Date of Prediction'].min(),
                'latest_prediction': event_data['Date of Prediction'].max(),
                'earliest_event': event_data['Predicted Event Date'].min(),
                'latest_event': event_data['Predicted Event Date'].max()
            }

        return stats

    def calculate_linear_regression(self, event_data):
        """Calculate linear regression for prediction timeline."""
        # Convert dates to numeric values for regression
        
        x_numeric = event_data['Date of Prediction'].astype(int)
        y_numeric = event_data['Predicted Event Date'].astype(int)

        # Calculate linear regression
        if len(x_numeric) > 1:
            coefficients = np.polyfit(x_numeric, y_numeric, 1)
            slope = coefficients[0]
            intercept = coefficients[1]

            # Extend line to intersect with real-time reference (where x == y)
            # Find intersection point: y = slope*x + intercept intersects y = x
            # So: x = slope*x + intercept => x - slope*x = intercept => x(1-slope) = intercept => x = intercept/(1-slope)

            x_min = x_numeric.min()
            x_max = x_numeric.max()

            # Always extend to real-time intersection and beyond data range
            if abs(1 - slope) > 1e-10:  # Avoid division by zero
                intersection_x = intercept / (1 - slope)

                # Create a range that includes both data and extends to/ beyond intersection
                data_range = x_max - x_min
                extension_distance = max(data_range * 0.1, 30)  # Extend by 30% of data or 30 days minimum

                # Ensure we include the intersection point
                x_extended_min = x_min - extension_distance
                x_extended_max = max(x_max, intersection_x) + extension_distance

            else:
                # If slope is very close to 1, just extend the data range
                x_extended_min = x_min - abs(x_max - x_min) * 0.3
                x_extended_max = x_max + abs(x_max - x_min) * 0.3

            x_line = np.linspace(x_extended_min, x_extended_max, 10)
            y_line = slope * x_line + intercept

            # Convert back to dates
            x_dates = pd.to_datetime(x_line)
            y_dates = pd.to_datetime(y_line)

            return x_dates, y_dates, slope, intercept
        else:
            # Not enough data points for regression
            return None, None, None, None

    def create_timeline_chart(self, event_name=None):
        """Create timeline visualization showing prediction evolution."""
        if event_name:
            plot_data = self.filter_by_event(event_name)
            title = f'Prediction Timeline: {event_name}'
        else:
            plot_data = self.data
            title = 'Prediction Timeline: All Events'

        fig = go.Figure()

        # Calculate date range for reference lines
        min_date = min(plot_data['Date of Prediction'].min(), plot_data['Predicted Event Date'].min())
        max_date = max(plot_data['Date of Prediction'].max(), plot_data['Predicted Event Date'].max())

        # Generate dynamic color mapping for events
        events = plot_data['Event'].unique()
        color_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]

        event_colors = {}
        for i, event in enumerate(events):
            event_colors[event] = color_palette[i % len(color_palette)]

        for event in events:
            event_data = plot_data[plot_data['Event'] == event]
            event_color = event_colors[event]

            # Add scatter points for this event
            fig.add_trace(go.Scatter(
                x=event_data['Date of Prediction'],
                y=event_data['Predicted Event Date'],
                mode='markers',
                name=event,
                marker=dict(
                    color=event_color,
                    size=8
                ),
                hovertemplate=
                '<b>%{fullData.name}</b><br>' +
                'Prediction Date: %{x}<br>' +
                'Predicted Event Date: %{y}<br>' +
                'Prediction Window: %{customdata} days<extra></extra>',
                customdata=event_data['Prediction Window']
            ))

            # Calculate and add linear regression trend line
            x_regression, y_regression, slope, intercept = self.calculate_linear_regression(event_data)
            if x_regression is not None and y_regression is not None:
                min_date = min(min_date, x_regression.min())
                max_date = max(max_date, x_regression.max())
                # Use the same color as the event markers
                event_color = event_colors[event]

                fig.add_trace(go.Scatter(
                    x=x_regression,
                    y=y_regression,
                    mode='lines',
                    name=f'{event} Trend',
                    line=dict(
                        width=3,
                        dash='solid',
                        color=event_color
                    ),
                    opacity=0.8,
                    hovertemplate=
                    f'<b>{event} Trend Line</b><br>' +
                    'Slope: %{customdata:.4f}<br>' +
                    'Prediction Date: %{x}<br>' +
                    'Predicted Event Date: %{y}<extra></extra>',
                    customdata=np.full(len(x_regression), slope),
                    showlegend=True
                ))

        # Add real-time reference line (where prediction date = predicted event date)
        fig.add_trace(go.Scatter(
            x=[min_date, max_date],
            y=[min_date, max_date],
            mode='lines',
            name='Real Time',
            line=dict(
                color='rgba(128, 128, 128, 0.5)',
                width=2,
                dash='dash'
            ),
            hovertemplate='Real Time Reference<br>Date: %{x}<extra></extra>',
            showlegend=True
        ))

        # Add "Today" vertical reference line
        today = datetime.now()
        fig.add_trace(go.Scatter(
            x=[today, today],
            y=[min_date, max_date],
            mode='lines',
            name='Last Updated',
            line=dict(
                color='rgba(255, 0, 0, 0.8)',  # Orange color for visibility
                width=3,
                dash='dot'
            ),
            hovertemplate='Today<br>Date: %{x}<extra></extra>',
            showlegend=True
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Date of Prediction',
            yaxis_title='Predicted Event Date',
            hovermode='closest',
            template='plotly_white'
        )

        return fig.to_html(full_html=False, include_plotlyjs=True)

    def create_prediction_window_chart(self, event_name=None):
        """Create chart showing prediction windows over time."""
        if event_name:
            plot_data = self.filter_by_event(event_name)
            title = f'Prediction Windows: {event_name}'
        else:
            plot_data = self.data
            title = 'Prediction Windows: All Events'

        fig = go.Figure()

        for event in plot_data['Event'].unique():
            event_data = plot_data[plot_data['Event'] == event]

            fig.add_trace(go.Scatter(
                x=event_data['Date of Prediction'],
                y=event_data['Prediction Window'],
                mode='markers',
                name=event,
                hovertemplate=
                '<b>%{fullData.name}</b><br>' +
                'Prediction Date: %{x}<br>' +
                'Window: %{y} days<extra></extra>'
            ))

        fig.update_layout(
            title=title,
            xaxis_title='Date of Prediction',
            yaxis_title='Prediction Window (days)',
            hovermode='closest',
            template='plotly_white'
        )

        return fig.to_html(full_html=False, include_plotlyjs=False)

    def create_statistics_chart(self):
        """Create statistical summary visualization."""
        stats = self.calculate_event_statistics()

        events = list(stats.keys())
        avg_windows = [stats[event]['avg_prediction_window'] for event in events]
        counts = [stats[event]['count'] for event in events]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Average Prediction Windows by Event', 'Prediction Count by Event'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )

        fig.add_trace(
            go.Bar(x=events, y=avg_windows, name='Avg Window (days)'),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(x=events, y=counts, name='Count'),
            row=1, col=2
        )

        fig.update_layout(
            title='Statistical Summary by Event',
            template='plotly_white',
            showlegend=False
        )

        return fig.to_html(full_html=False, include_plotlyjs=False)


class HTMLGenerator:
    """Generates the final HTML output with embedded charts."""

    def __init__(self, processor):
        self.processor = processor
        self.template = self._get_html_template()

    def _get_html_template(self):
        """Get the HTML template from file."""
        template_path = Path(__file__).parent / 'templates' / 'base.html'
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Template file not found: {template_path}")

    def generate_html(self):
        """Generate the complete HTML output."""
        events = self.processor.get_unique_events()
        stats = self.processor.calculate_event_statistics()
        generation_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Create charts
        timeline_chart = self.processor.create_timeline_chart()
        window_chart = self.processor.create_prediction_window_chart()
        stats_chart = self.processor.create_statistics_chart()

        # Render template
        template = Template(self.template)
        html_output = template.render(
            events=events,
            event_stats=stats,
            timeline_chart=timeline_chart,
            window_chart=window_chart,
            stats_chart=stats_chart,
            generation_date=generation_date
        )

        return html_output


def main():
    """Main function to generate the visualization."""
    print("🚀 Event Prediction Visualization Generator")
    print("=" * 50)

    # Initialize processor with sample data
    processor = PredictionDataProcessor('Example_data.csv')
    processor.load_data()

    # Generate HTML
    generator = HTMLGenerator(processor)
    html_output = generator.generate_html()

    # Write to file
    output_file = 'generated/event_prediction_analysis.html'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_output)

    print(f"✅ Generated {output_file}")
    print("📊 Key Statistics:")
    all_stats = processor.calculate_event_statistics()
    for event, stats in all_stats.items():
        print(f"  • {event}: {stats['count']} predictions, avg window: {stats['avg_prediction_window']:.0f} days")

    print(f"\n🌐 Open {output_file} in your browser to view the visualization!")


if __name__ == '__main__':
    main()