# 🔮 Event Prediction Visualization Tool

A Python-based tool for creating interactive HTML visualizations from event prediction data. This tool processes CSV files containing prediction data and generates beautiful, interactive charts showing prediction timelines, accuracy windows, and statistical summaries.

## ✨ Features

- **Interactive Timeline Charts**: Visualize individual predictions as scatter points with real-time reference line and linear regression trend lines
- **Prediction Window Analysis**: Track the time windows between predictions and actual events
- **Statistical Summaries**: View key statistics including slip rate trends for each event type
- **Modern UI**: Clean, responsive web interface with hover effects and smooth interactions
- **Multiple Event Support**: Handle multiple event types in a single visualization
- **Export Ready**: Generates standalone HTML files that can be shared or hosted anywhere

## 📋 Prerequisites

- Python 3.10 or higher
- Poetry (for dependency management)

## 🚀 Installation

1. **Clone or download the project files**

2. **Install dependencies using Poetry:**
   ```bash
   poetry install
   ```

   This will install all required packages:
   - pandas (data processing)
   - plotly (interactive charts)
   - jinja2 (HTML templating)

## 📖 Usage

### Basic Usage

1. **Prepare your data**: Create a CSV file with prediction data (see [Input Data Format](#input-data-format) below)

2. **Run the visualization generator:**
   ```bash
   python3 generate_prediction_viz.py
   ```

3. **View the results**: Open `generated/event_prediction_analysis.html` in your web browser

### Customizing Input Data

You can modify the CSV file path in the script by editing line 477 in `generate_prediction_viz.py`:
```python
processor = PredictionDataProcessor('your_data_file.csv')
```

## 📊 Input Data Format

The tool expects a CSV file with the following columns:

| Column | Description | Format | Example |
|--------|-------------|---------|---------|
| Event | Name/type of the predicted event | Text | `my_test_event` |
| Date of Prediction | When the prediction was made | MM/DD/YYYY | `1/15/2025` |
| Predicted Event Date | When the event is predicted to occur | MM/DD/YYYY | `6/20/2025` |
| Tag | Optional category or tag for the prediction | Text | `important` |

### Example CSV Structure:
```csv
Event,Date of Prediction,Predicted Event Date,Tag
my_test_event,1/1/2025,6/1/2025,
my_test_event,1/7/2025,6/3/2025,
my_test_event_2,1/1/2025,7/1/2025,
```

## 🎯 Output

The tool generates a single HTML file containing:

- **Prediction Timeline**: Scatter plot showing individual predictions as points with linear regression trend lines and a real-time reference line (diagonal where x=y) for comprehensive trend analysis
- **Statistics Overview**: Bar charts showing slip rate (days/day) trends and prediction counts by event
- **Prediction Windows**: Visualization of time windows between predictions and events
- **Statistical Summary**: Bar charts showing average prediction windows and prediction counts by event
- **Interactive Controls**: Dropdown to filter by specific events
- **Responsive Design**: Works on desktop and mobile devices

### Generated File Location
- Output file: `generated/event_prediction_analysis.html`
- File size: Typically 4-5 MB (includes interactive Plotly charts)

## 🏗️ Project Structure

```
sliplaunch/
├── generate_prediction_viz.py    # Main visualization generator script
├── Example_data.csv             # Sample input data
├── pyproject.toml              # Poetry project configuration
├── poetry.lock                 # Dependency lock file
├── generated/                  # Output directory (created automatically)
│   └── event_prediction_analysis.html
└── README.md                   # This file
```

## 📈 What It Does

1. **Data Processing**: Loads and validates CSV prediction data
2. **Statistical Analysis**: Calculates prediction windows, averages, and other metrics
3. **Chart Generation**: Creates interactive Plotly visualizations
4. **HTML Templating**: Uses Jinja2 to generate a modern, responsive web interface
5. **File Output**: Saves everything to a standalone HTML file

## 🔧 Customization

### Modifying the Visualization

The HTML template and styling can be customized in the `HTMLGenerator` class within `generate_prediction_viz.py`. Key customization areas:

- **Colors**: Modify the CSS variables in the HTML template
- **Chart Types**: Add new chart types by extending the processor classes
- **Layout**: Adjust the responsive grid layout in the CSS

### Adding New Metrics

You can extend the `PredictionDataProcessor` class to calculate additional statistics:

```python
def calculate_custom_metric(self):
    # Add your custom calculation here
    pass
```

## 🤝 Contributing

To contribute to this project:

1. Make sure you have Poetry installed
2. Create a feature branch
3. Install development dependencies: `poetry install`
4. Make your changes
5. Test with sample data
6. Submit a pull request

## 📄 License

This project is part of the sliplaunch toolkit for event prediction analysis.

---

**Generated on**: 2025-09-28
**Version**: 0.1.0