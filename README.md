# AI Act Processor

## Overview
The AI Act Processor is a tool designed to analyze and generate compliance reports for various AI-related projects. It evaluates compliance with the AI Act by assessing key categories such as risk management, data governance, technical documentation, and more. The tool processes input data, generates visualizations, and provides detailed compliance reports.

## Features
- **Compliance Analysis**: Evaluates AI projects against the AI Act's requirements.
- **Report Generation**: Produces detailed HTML reports for each project.
- **Visualization**: Generates charts and graphs to summarize compliance scores.
- **Data Processing**: Includes scripts for analyzing and processing input data.

## Project Structure
```
AIActProcessor/
├── reports/                     # Generated compliance reports
├── reports_raw_download/        # Raw downloaded reports
├── checklist_analysis.py        # Script for analyzing compliance checklists
├── pdf_analyzer.py              # Script for processing PDF documents
├── rag_pdf_analyzer.py          # Script for RAG-based PDF analysis
├── process_cpv_and_plot.py      # Script for processing CPV data and generating plots
├── main_script.py               # Main script for running the analysis
├── compliance_summary.csv       # Summary of compliance scores
├── compliance_groups_bar.png    # Bar chart summarizing compliance groups
├── histograms/                  # Histograms for various compliance categories
└── README.md                    # Project documentation
```

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd AIActProcessor
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Download the reports using the scraping_code.py script. This script is by default downloading open access articles from
arxiv and scopus. To download from scopus you need to create an API key (https://dev.elsevier.com/) and modify the line in the script that refers to that  
   ```bash
   api_key = "YOUR_ELSEVIER_API_KEY"  # Replace with your actual API key 
   ```
   The script can be run and with the default settings it will
   attempt to download 200 papers from 2025 to 2023 using the query "AI geotechnical engineering".

2. Run the main script to process the data and generate reports:
   ```bash
   python main_script.py
   ```
3. View the generated reports in the `reports/` directory.

## Key Scripts
- **`main_script.py`**: Orchestrates the entire analysis process.
- **`checklist_analysis.py`**: Analyzes compliance checklists.
- **`pdf_analyzer.py`**: Processes PDF documents for compliance data.
- **`process_csv_and_plot.py`**: Processes CPV data and generates visualizations.

## Outputs
- **HTML Reports**: Detailed compliance reports for each project.
- **CSV Summaries**: Aggregated compliance scores.
- **Visualizations**: Charts and graphs summarizing compliance metrics.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

## Contact
For questions or support, please contact [eleni.smyrniou@deltares.nl].
