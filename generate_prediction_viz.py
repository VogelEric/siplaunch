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
import argparse
from pandas._libs.tslibs.np_datetime import OutOfBoundsTimedelta


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
        """Calculate statistical measures and linear regression for events."""
        if event_name:
            filtered_data = self.filter_by_event(event_name)
        else:
            filtered_data = self.data

        stats = {}
        for event in filtered_data['Event'].unique():
            event_data = filtered_data[filtered_data['Event'] == event]

            # Basic statistics
            basic_stats = {
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

            # Linear regression for trend analysis
            regression_data = self.calculate_event_trend(event_data)

            # Combine basic stats with regression data
            stats[event] = {**basic_stats, **regression_data}

        return stats

    def calculate_event_trend(self, event_data):
        """Calculate comprehensive linear regression trend analysis for event data."""
        # Sort data by prediction date for progressive analysis
        event_data = event_data.sort_values('Date of Prediction')
        x_numeric = event_data['Date of Prediction'].astype(int)
        y_numeric = event_data['Predicted Event Date'].astype(int)

        n_points = len(x_numeric)
        progressive_slopes = []
        progressive_intercepts = []
        progressive_sizes = []
        progressive_x = []
        progressive_y = []

        # Calculate progressive best fit lines from 3 points to full dataset
        if n_points >= 3:
            for subset_size in range(3, n_points + 1):
                # Take the first 'subset_size' points for progressive analysis
                x_subset = x_numeric.iloc[:subset_size]
                y_subset = y_numeric.iloc[:subset_size]

                coefficients = np.polyfit(x_subset, y_subset, 1)
                slope = coefficients[0]
                intercept = coefficients[1]

                intersect = None
                if (1-slope) > 1e-10:
                    intersect = intercept / (1-slope)

                progressive_slopes.append(slope)
                progressive_intercepts.append(intercept)
                progressive_sizes.append(subset_size)
                progressive_x.append(pd.to_datetime(x_subset.iloc[-1]))
                progressive_y.append(pd.to_datetime(intersect))

            # Calculate final best fit on complete dataset
            final_coefficients = np.polyfit(x_numeric, y_numeric, 1)
            final_slope = final_coefficients[0]
            final_intercept = final_coefficients[1]

            # Extend final line to intersect with real-time reference
            x_min, x_max = x_numeric.min(), x_numeric.max()

            if abs(1 - final_slope) > 1e-10:
                intersection_x = final_intercept / (1 - final_slope)
                data_range = x_max - x_min
                extension_distance = max(data_range * 0.1, 30)
                x_extended_min = x_min - extension_distance
                x_extended_max = max(x_max, intersection_x) + extension_distance
            else:
                x_extended_min = x_min - abs(x_max - x_min) * 0.3
                x_extended_max = x_max + abs(x_max - x_min) * 0.3

            x_line = np.linspace(x_extended_min, x_extended_max, 10)
            y_line = final_slope * x_line + final_intercept
            x_dates = pd.to_datetime(x_line)
            y_dates = pd.to_datetime(y_line)

            # Calculate intersection date
            intersection_date = progressive_y[-1]

            return {
                'trend_slope': final_slope,
                'trend_intercept': final_intercept,
                'trend_x_dates': x_dates,
                'trend_y_dates': y_dates,
                'realtime_intersection': intersection_date,
                'progressive_slopes': progressive_slopes,
                'progressive_intercepts': progressive_intercepts,
                'progressive_sizes': progressive_sizes,
                'progressive_x': progressive_x,
                'progressive_y': progressive_y,
                'final_data_points': n_points,
                'has_trend': True
            }
        else:
            return {
                'trend_slope': None,
                'trend_intercept': None,
                'trend_x_dates': None,
                'trend_y_dates': None,
                'realtime_intersection': None,
                'progressive_slopes': None,
                'progressive_intercepts': None,
                'progressive_sizes': None,
                'progressive_x': None,
                'progressive_y': None,
                'final_data_points': n_points,
                'has_trend': False
            }

    def calculate_weighted_overall_slip_rate(self):
        """Calculate weighted overall slip rate across all events."""
        stats = self.calculate_event_statistics()
        total_predictions = 0
        weighted_sum = 0

        for event, data in stats.items():
            if data['has_trend'] and data['trend_slope'] is not None:
                count = data['count']
                slope = data['trend_slope']
                total_predictions += count
                weighted_sum += slope * count

        if total_predictions > 0:
            return weighted_sum / total_predictions
        else:
            return None


    def create_timeline_chart(self, event_name=None):
        """Create timeline visualization showing prediction evolution."""
        if event_name:
            plot_data = self.filter_by_event(event_name)
            title = f'Prediction Timeline: {event_name}'
        else:
            plot_data = self.data
            title = 'Prediction Timeline: All Events'

        # Get statistics with pre-calculated regression data
        event_stats = self.calculate_event_statistics(event_name)

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

            # Add progressive trend lines using pre-calculated data
            if event_stats[event]['has_trend']:
                trend_data = event_stats[event]

                # Add initial trend line (first 3 points) - faded to show starting point
                if (trend_data['progressive_sizes'] and len(trend_data['progressive_sizes']) > 0):
                    initial_slope = trend_data['progressive_slopes'][0]
                    initial_intercept = trend_data['progressive_intercepts'][0]
                    progressive_x = trend_data['progressive_x']
                    progressive_y = trend_data['progressive_y']

                    # Create line for initial trend
                    x_min_numeric, x_max_numeric = event_data['Date of Prediction'].astype(int).min(), event_data['Date of Prediction'].astype(int).max()
                    x_initial_line = np.linspace(x_min_numeric, x_max_numeric, 10)
                    y_initial_line = initial_slope * x_initial_line + initial_intercept
                    x_initial_dates = pd.to_datetime(x_initial_line)
                    y_initial_dates = pd.to_datetime(y_initial_line)

                    #print(progressive_y)

                    fig.add_trace(go.Scatter(
                        x=progressive_x,
                        y=progressive_y,
                        mode='lines',
                        name=f'{event} Intercept Dates',
                        line=dict(
                            width=3,
                            dash='solid',
                            color=event_color
                        ),
                        opacity=0.8,
                        hovertemplate=
                        f'<b>{event} Initial Trend (3 points)</b><br>' +
                        'Slope: %{customdata:.4f}<br>' +
                        'Prediction Date: %{x}<br>' +
                        'Predicted Event Date: %{y}<extra></extra>',
                        customdata=np.full(len(x_initial_dates), initial_slope),
                        showlegend=False
                    ))

                # Add final trend line (complete dataset) - solid to show current analysis
                x_regression = trend_data['trend_x_dates']
                y_regression = trend_data['trend_y_dates']
                slope = trend_data['trend_slope']

                # Update date range to include trend line extensions
                if x_regression is not None and y_regression is not None:
                    min_date = min(min_date, x_regression.min())
                    max_date = max(max_date, x_regression.max())

                fig.add_trace(go.Scatter(
                    x=x_regression,
                    y=y_regression,
                    mode='lines',
                    name=f'{event} Final Trend',
                    line=dict(
                        width=2,
                        dash='dot',
                        color=event_color
                    ),
                    opacity=0.4,
                    hovertemplate=
                    f'<b>{event} Final Trend (' + str(trend_data["final_data_points"]) + ' points)</b><br>' +
                    'Slope: %{customdata:.4f}<br>' +
                    'Prediction Date: %{x}<br>' +
                    'Predicted Event Date: %{y}<extra></extra>',
                    customdata=np.full(len(x_regression), slope),
                    showlegend=False
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
            template='plotly_white',
            height=800  # Increased height for better visibility
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
        slip_rates = [stats[event]['trend_slope'] for event in events]
        counts = [stats[event]['count'] for event in events]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Slip Rate (days/day) by Event', 'Prediction Count by Event'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )

        fig.add_trace(
            go.Bar(x=events, y=slip_rates, name='Slip Rate (days/day)'),
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

    def create_prediction_summary_table(self):
        """Create summary table data for all events."""
        stats = self.calculate_event_statistics()

        # Create table data with all required fields
        table_data = []
        for event, data in stats.items():
            table_data.append({
                'event_name': event,
                'most_recent_prediction': data['latest_prediction'],
                'intercept_date': data['realtime_intersection'],
                'prediction_count': data['count'],
                'avg_slip_rate': data['trend_slope']
            })

        # Sort by most recent prediction date (newest first)
        table_data.sort(key=lambda x: x['most_recent_prediction'], reverse=True)

        return table_data

    def get_serializable_data(self):
        """Get raw data in a format suitable for JavaScript embedding."""
        # Convert data to JSON-serializable format
        data_list = []
        for _, row in self.data.iterrows():
            data_list.append({
                'event': row['Event'],
                'prediction_date': row['Date of Prediction'].strftime('%Y-%m-%d'),
                'predicted_event_date': row['Predicted Event Date'].strftime('%Y-%m-%d'),
                'prediction_window': int(row['Prediction Window']),
                'tag': row.get('Tag', '') if 'Tag' in row else ''
            })

        return {
            'raw_data': data_list,
            'events': sorted(self.data['Event'].unique().tolist()),
            'date_range': {
                'min': self.data['Date of Prediction'].min().strftime('%Y-%m-%d'),
                'max': self.data['Date of Prediction'].max().strftime('%Y-%m-%d')
            }
        }


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

        # Get serializable data for JavaScript embedding
        embedded_data = self.processor.get_serializable_data()

        # Create charts
        timeline_chart = self.processor.create_timeline_chart()
        window_chart = self.processor.create_prediction_window_chart()
        stats_chart = self.processor.create_statistics_chart()

        # Create prediction summary table data
        summary_table = self.processor.create_prediction_summary_table()

        # Render template
        template = Template(self.template)
        html_output = template.render(
            events=events,
            event_stats=stats,
            timeline_chart=timeline_chart,
            window_chart=window_chart,
            stats_chart=stats_chart,
            summary_table=summary_table,
            generation_date=generation_date,
            embedded_data=embedded_data
        )

        return html_output


def main(csv_file=None):
    """Main function to generate the visualization.

    Args:
        csv_file (str, optional): Path to CSV file containing prediction data.
                                 Defaults to 'Example_data.csv' if not provided.
    """
    print("🚀 Event Prediction Visualization Generator")
    print("=" * 50)

    # Use provided CSV file or default to Example_data.csv
    if csv_file is None:
        csv_file = 'Example_data.csv'

    if not os.path.exists(csv_file):
        print(f"❌ Error: CSV file '{csv_file}' not found!")
        print("   Please ensure the file exists or provide a valid path.")
        return

    # Initialize processor with specified data file
    processor = PredictionDataProcessor(csv_file)
    processor.load_data()
    overall_slip_rate = processor.calculate_weighted_overall_slip_rate()

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

    # Display individual event statistics
    for event, stats in all_stats.items():
        progressive_info = ""
        if stats['has_trend'] and stats['progressive_sizes']:
            n_subsets = len(stats['progressive_sizes'])
            progressive_info = f" (analyzed {n_subsets} progressive subsets)"

        print(f"  • {event}: {stats['count']} predictions, slip rate: {stats['trend_slope']:.3f} days/day{progressive_info}")

    # Display weighted overall slip rate
    if overall_slip_rate is not None:
        print(f"  • Overall (weighted): {overall_slip_rate:.3f} days/day")
    else:
        print("  • Overall: Unable to calculate weighted slip rate")

    print(f"\n🌐 Open {output_file} in your browser to view the visualization!")


if __name__ == '__main__':
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description='Generate event prediction visualization from CSV data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python generate_prediction_viz.py                    # Use default Example_data.csv
  python generate_prediction_viz.py -f my_data.csv     # Use custom CSV file
  python generate_prediction_viz.py --file data.csv    # Use custom CSV file
  python generate_prediction_viz.py --help            # Show this help message
        """
    )

    parser.add_argument(
        '-f', '--file',
        type=str,
        default=None,
        help='Path to CSV file containing prediction data (default: Example_data.csv)'
    )

    parser.add_argument(
        '-v', '--version',
        action='version',
        version='Event Prediction Visualization Generator v0.2.0'
    )

    # Parse arguments
    args = parser.parse_args()

    # Run main function with provided arguments
    main(args.file)