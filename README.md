# 🔮 Event Prediction Visualization Tool

A Python-based tool for creating interactive HTML visualizations from event prediction data. This tool processes CSV files containing prediction data and generates beautiful, interactive charts showing prediction timelines, accuracy windows, and statistical summaries.

## ✨ Features

- **📈 Advanced Trend Analysis**: Linear regression with real-time intersection calculations showing when predictions would become "perfect"
- **🎨 Dynamic Multi-Event Visualization**: Automatic color-coded visualization supporting unlimited event types with professional palette
- **📊 Comprehensive Summary Table**: Sortable table with event names, recent predictions, slip rates, counts, and intercept dates
- **📅 Current Date Reference**: "Today" vertical line providing immediate temporal context
- **🔬 Slip Rate Analytics**: Replace simple averages with trend slope analysis (days/day) for actionable insights
- **🏗️ Professional Architecture**: Separated HTML templates for maintainability and customization
- **📱 Responsive Design**: Modern, mobile-friendly interface with gradient styling and hover effects
- **⚡ Real-time Processing**: Fast analysis of large datasets with robust error handling
- **📋 Interactive Controls**: Dropdown filtering and responsive table sorting
- **🎯 Professional Export**: Standalone HTML files with embedded analytics and interactive charts

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
   # Use default CSV file (Example_data.csv)
   python3 generate_prediction_viz.py

   # Or specify a custom CSV file
   python3 generate_prediction_viz.py -f your_data.csv
   ```

3. **View the results**: Open `generated/event_prediction_analysis.html` in your web browser

### Command Line Options

The visualization generator supports several command line options:

```bash
python3 generate_prediction_viz.py [OPTIONS]

Options:
  -f, --file FILE     Path to CSV file containing prediction data
                     (default: Example_data.csv)
  -v, --version       Show program version
  -h, --help          Show help message and exit

Examples:
  python3 generate_prediction_viz.py                    # Use default file
  python3 generate_prediction_viz.py -f my_data.csv     # Use custom file
  python3 generate_prediction_viz.py --file data.csv    # Use custom file
```

### Customizing Input Data

You can specify a custom CSV file when running the generator:
```bash
python3 generate_prediction_viz.py -f your_data_file.csv
```

Or get help on all available options:
```bash
python3 generate_prediction_viz.py --help
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

The tool generates a comprehensive HTML dashboard containing:

### 📈 Prediction Timeline
- **Individual Predictions**: Scatter plot showing each prediction as a distinct point
- **Trend Lines**: Linear regression lines showing accuracy direction over time
- **Real-time Reference**: Diagonal line (y=x) showing perfect prediction timing
- **Current Date**: Vertical orange line showing today's date
- **Interactive Tooltips**: Hover for detailed prediction information and slope data

### 📊 Statistics Overview
- **Slip Rate Analysis**: Bar chart showing rate of prediction accuracy change (days/day)
- **Prediction Volume**: Count of predictions per event type
- **Visual Indicators**: Color-coded trends (positive = getting worse, negative = improving)

### 📋 Prediction Summary Table
- **Event Names**: Bold, emphasized with gray background for easy identification
- **Most Recent Predictions**: Chronologically sorted (newest first) for priority awareness
- **Average Slip Rates**: Trend direction and magnitude with 3-decimal precision
- **Prediction Counts**: Volume tracking per event
- **Intercept Dates**: When trends would reach real-time accuracy (highlighted with warm background)
- **Professional Styling**: Responsive table with hover effects and column emphasis

### 📅 Prediction Windows
- **Time Window Tracking**: Evolution of prediction accuracy windows over time
- **Event Comparison**: Side-by-side analysis of different event types
- **Interactive Markers**: Hover tooltips showing detailed window information

### Generated File Location
- Output file: `generated/event_prediction_analysis.html`
- File size: Typically 4-5 MB (includes interactive Plotly charts)

## 🏗️ Project Structure

```
sliplaunch/
├── generate_prediction_viz.py    # Main application script with comprehensive analytics
├── templates/
│   └── base.html               # HTML template with modern styling and responsive design
├── Example_data.csv            # Sample input data with multiple event types
├── generated/                  # Output directory (auto-created)
│   └── event_prediction_analysis.html
├── pyproject.toml             # Poetry configuration with all dependencies
├── poetry.lock               # Dependency lock file
└── README.md                 # This comprehensive documentation
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

## 🚀 Future Improvements

- **🔄 Interactive Filtering**: Real-time chart updates when selecting events from dropdown
- **📊 Additional Chart Types**: Prediction accuracy distributions and confidence intervals
- **📈 Comparative Analysis**: Side-by-side event comparison tools
- **🎛️ Configuration Files**: External configuration for colors, thresholds, and display options
- **📤 Export Options**: PDF, PNG, and Excel export capabilities
- **🗂️ Data Validation**: Comprehensive CSV validation with error reporting
- **📋 Batch Processing**: Support for multiple CSV files with combined analysis

## 📄 License

This project is part of the sliplaunch toolkit for event prediction analysis.

---

**Generated on**: 2025-09-28
**Version**: 0.2.0
**Dataset Capacity**: Unlimited events with automatic scaling