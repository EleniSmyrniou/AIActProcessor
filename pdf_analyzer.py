import os
import json
import pandas as pd
import PyPDF2
import requests
from tqdm import tqdm
from ollama_api import OllamaAPI  # Import the OllamaAPI class

class AIActComplianceAnalyzer:
    def __init__(self, ollama_base_url="http://localhost:11434", model_name="response_qwen3"):
        """
        Initialize the AI Act Compliance Analyzer
        
        Args:
            ollama_base_url: URL where Ollama is running
            model_name: Name of the Ollama model to use (e.g., llama3, mistral, etc.)
        """
        self.ollama_api = OllamaAPI(url=ollama_base_url, model=model_name)  # Use OllamaAPI
        self.checklist = self._load_ai_act_checklist()
        
    def _load_ai_act_checklist(self):
        """Load the AI Act checklist categories and items"""
        return {
            "Risk Management": [
                "Identify risks to health, safety, and fundamental rights",
                "Analyze & document each risk",
                "Define mitigation measures for each identified risk",
                "Test model under representative real-world conditions",
                "Record test results and confirm effectiveness of mitigation measures",
                "Plan ongoing adaptation: schedule periodic risk reviews during deployment"
            ],
            "Data Governance & Bias": [
                "Inventory datasets (training/validation/test)",
                "Document data origin and collection methods",
                "Record preprocessing steps",
                "List assumptions made about data",
                "Identify data gaps",
                "Assess and document biases in data or labels",
                "Ensure representativeness of data"
            ],
            "Technical Documentation & Transparency": [
                "State intended purpose and scope of the AI tool",
                "List responsible entity (provider contact)",
                "Define intended users",
                "Specify known limitations and prohibited use cases",
                "Publish performance metrics",
                "Describe model architecture and training regimen",
                "Document human-in-the-loop measures",
                "Outline resource requirements",
                "Specify model lifetime and maintenance plan"
            ],
            "Accuracy, Robustness & Cybersecurity": [
                "Benchmark accuracy against conventional methods",
                "Test robustness on edge cases",
                "Implement fail-safes",
                "Assess adversarial vulnerability",
                "Secure data pipeline against tampering"
            ],
            "Logging & Traceability": [
                "Automate event logging for every model invocation",
                "Log model version and random seed",
                "Flag high-risk events in logs",
                "Store logs securely"
            ],
            "Human Oversight & Governance": [
                "Define review workflow",
                "Train users on tool limitations",
                "Maintain audit trail of human interventions"
            ],
            "Monitoring & Maintenance": [
                "Set up post-deployment monitoring",
                "Define retraining triggers",
                "Schedule periodic re-validation",
                "Update documentation after retraining",
                "Review cybersecurity posture regularly"
            ]
        }
    
    def extract_text_from_pdf(self, pdf_path):
        """
        Extract all text from a PDF document
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            str: Extracted text from the PDF
        """
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
            return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def query_ollama(self, prompt, max_tokens=4000):
        """
        Send a prompt to the Ollama API and get a response
        
        Args:
            prompt: The text prompt to send to the model
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            str: The model's response
        """
        try:
            response = self.ollama_api.send_test_message(prompt=prompt)
            if response and 'response' in response:
                return response['response']
            else:
                print("Error: No valid response from Ollama API")
                return ""
        except Exception as e:
            print(f"Exception when calling Ollama API: {e}")
            return ""
    
    def analyze_compliance(self, pdf_path, publication_metadata=None):
        """
        Analyze a publication's compliance with the AI Act checklist
        
        Args:
            pdf_path: Path to the PDF file
            publication_metadata: Dictionary containing metadata (title, authors, year)
            
        Returns:
            dict: Compliance scores and analysis for each checklist category
        """
        # Extract text from PDF
        pdf_text = self.extract_text_from_pdf(pdf_path)
        if not pdf_text:
            return {"error": "Failed to extract text from PDF"}
        
        # Prepare metadata string
        metadata_str = ""
        if publication_metadata:
            metadata_str = f"Title: {publication_metadata.get('title', 'Unknown')}\n"
            metadata_str += f"Authors: {publication_metadata.get('authors', 'Unknown')}\n"
            metadata_str += f"Year: {publication_metadata.get('year', 'Unknown')}\n\n"
        
        results = {}
        
        # Analyze each category separately to avoid context length issues
        for category, items in tqdm(self.checklist.items(), desc="Analyzing categories"):
            # Create the analysis prompt for this category
            prompt = f"""
            You are an expert in AI ethics and regulation. Your task is to analyze a research publication
            in geotechnics AI and determine to what extent it adheres to the AI Act requirements
            in the category of "{category}".
            
            {metadata_str}
            
            Here are the checklist items for {category}:
            {json.dumps(items, indent=2)}
            
            Below is the text from the publication. Analyze it and determine if it addresses each of the checklist items.
            Rate each item on a scale of 0-5:
            0: Not mentioned at all
            1: Briefly mentioned but no details
            2: Some details but inadequate
            3: Moderately addressed
            4: Well addressed
            5: Comprehensively addressed with best practices
            
            Publication text:
            {pdf_text[:50000]}  # Limit text to avoid context length issues
            
            Provide your analysis in JSON format with the following structure:
            {{"category": "{category}",
              "items": [
                {{"item": "Item 1", "score": 0-5, "justification": "Brief explanation"}},
                ...
              ],
              "average_score": 0.0,
              "summary": "Brief summary of compliance for this category"
            }}
            
            Return ONLY the valid JSON object.
            """
            
            # Query the model
            response = self.query_ollama(prompt)
            
            # Extract JSON from response
            try:
                # Find JSON object in the response
                json_str = response.strip()
                if not json_str.startswith('{'):
                    # Try to extract JSON if it's embedded in other text
                    import re
                    json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                    else:
                        raise ValueError("No JSON found in response")
                
                category_results = json.loads(json_str)
                results[category] = category_results
            except json.JSONDecodeError as e:
                results[category] = {
                    "category": category,
                    "error": f"Failed to parse model response: {e}",
                    "raw_response": response
                }
            except Exception as e:
                results[category] = {
                    "category": category,
                    "error": f"Error processing response: {e}",
                    "raw_response": response
                }
        
        return results
    
    def generate_compliance_report(self, results, output_path):
        """
        Generate a detailed compliance report in HTML format
        
        Args:
            results: Dictionary containing analysis results
            output_path: Path to save the HTML report
            
        Returns:
            str: Path to the generated report
        """
        # Calculate overall compliance score
        total_score = 0
        total_items = 0
        
        category_scores = {}
        for category, result in results.items():
            if "average_score" in result:
                category_scores[category] = result["average_score"]
                total_score += result["average_score"]
                total_items += 1
        
        overall_score = total_score / total_items if total_items > 0 else 0
        
        # Create HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Act Compliance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .summary {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                .category {{ margin-bottom: 30px; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .score-0 {{ background-color: #ffcccc; }}
                .score-1 {{ background-color: #ffddcc; }}
                .score-2 {{ background-color: #ffffcc; }}
                .score-3 {{ background-color: #e6ffcc; }}
                .score-4 {{ background-color: #ccffcc; }}
                .score-5 {{ background-color: #99ff99; }}
                .overall {{ font-size: 18px; font-weight: bold; }}
                .chart {{ width: 100%; height: 400px; margin-top: 30px; }}
            </style>
            <script type="text/javascript" src="https://www.gstatic.com/charts/loader.js"></script>
            <script type="text/javascript">
                google.charts.load('current', {{'packages':['corechart']}});
                google.charts.setOnLoadCallback(drawChart);
                
                function drawChart() {{
                    var data = google.visualization.arrayToDataTable([
                        ['Category', 'Compliance Score', {{ role: 'style' }}],
                        {', '.join([f"['{cat}', {score}, 'color: #3366cc']" for cat, score in category_scores.items()])}
                    ]);
                    
                    var options = {{
                        title: 'Compliance Scores by Category',
                        hAxis: {{ title: 'Category', titleTextStyle: {{ color: '#333' }} }},
                        vAxis: {{ minValue: 0, maxValue: 5, title: 'Score', titleTextStyle: {{ color: '#333' }} }},
                        legend: {{ position: 'none' }}
                    }};
                    
                    var chart = new google.visualization.ColumnChart(document.getElementById('chart_div'));
                    chart.draw(data, options);
                }}
            </script>
        </head>
        <body>
            <h1>AI Act Compliance Report</h1>
            
            <div class="summary">
                <h2>Executive Summary</h2>
                <p class="overall">Overall compliance score: {overall_score:.2f}/5.0</p>
                <p>This report analyzes compliance with the AI Act requirements across seven categories.</p>
            </div>
            
            <div id="chart_div" class="chart"></div>
        """
        
        # Add details for each category
        try:
            for category, result in results.items():
                html += f"""
                <div class="category">
                    <h2>{category}</h2>
                    <p>Average score: {result.get('average_score', 'N/A')}/5.0</p>
                    <p>{result.get('summary', '')}</p>

                    <table>
                        <tr>
                            <th>Checklist Item</th>
                            <th>Score</th>
                            <th>Justification</th>
                        </tr>
                """

                if "items" in result:
                    for item in result["items"]:
                        score = item.get('score', 0)
                        html += f"""
                        <tr class="score-{score}">
                            <td>{item.get('item', '')}</td>
                            <td>{score}/5</td>
                            <td>{item.get('justification', '')}</td>
                        </tr>
                        """

                html += """
                    </table>
                </div>
                """

            html += """
            </body>
            </html>
            """
        except Exception as e:
            print(f"Error generating HTML report: {e}")
            html += f"<p>Error generating report: {e}</p></body></html>"
        
        # Write HTML to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return output_path
    
    def batch_analyze_publications_with_direct_paths(self, publications_df, output_dir):
        """
        Analyze multiple publications using direct paths from the 'Document' column
        
        Args:
            publications_df: DataFrame with columns 'Document', 'Title', 'Authors', 'Year', etc.
            output_dir: Directory to save reports
            
        Returns:
            pd.DataFrame: Summary of results for all publications
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results_summary = []
        
        for _, row in tqdm(publications_df.iterrows(), total=len(publications_df), desc="Analyzing publications"):
            try:
                pdf_path = row['Document']

                if not os.path.exists(pdf_path) or not isinstance(pdf_path, str):
                    print(f"PDF file not found or invalid path: {pdf_path}")
                    continue
                
                # Extract filename for the report
                filename = os.path.basename(pdf_path)

                # Prepare metadata
                metadata = {
                    'title': row.get('Title', ''),
                    'authors': row.get('Authors', ''),
                    'year': row.get('Year', '')
                }

                print(f"Analyzing: {metadata['title']} ({pdf_path})")

                # Analyze compliance
                results = self.analyze_compliance(pdf_path, metadata)

                # Generate report
                report_filename = f"{os.path.splitext(filename)[0]}_compliance_report.html"
                report_path = os.path.join(output_dir, report_filename)
                self.generate_compliance_report(results, report_path)

                # Collect summary data
                summary = {
                    'filename': filename,
                    'title': metadata['title'],
                    'authors': metadata['authors'],
                    'year': metadata['year'],
                    'report_path': report_path
                }

                # Add category scores
                for category, result in results.items():
                    if "average_score" in result:
                        summary[f"{category}_score"] = result["average_score"]

                # Calculate overall score
                category_scores = [result.get("average_score", 0) for result in results.values() if "average_score" in result]
                summary['overall_score'] = sum(category_scores) / len(category_scores) if category_scores else 0

                results_summary.append(summary)
            except Exception as e:
                print(f"Error analyzing publication {row['filename']}: {e}")
                # write fake summary
                summary = {
                    'filename': row['filename'],
                    'title': row.get('Title', ''),
                    'authors': row.get('Authors', ''),
                    'year': row.get('Year', ''),
                    'report_path': '',
                    'overall_score': 0
                }
                results_summary.append(summary)
        
        return pd.DataFrame(results_summary)


# Example usage script
if __name__ == "__main__":
    # 1. Create sample publications DataFrame (replace with your actual data) Title,Authors,Year,Journal,URL,Abstract,Document
    publications_data = {
        'Document': ['SchemaGAN_ A conditional Generative Adversarial Network for geotechnical subsurface schematisation.pdf'],
        'Title': ['SchemaGAN: A conditional Generative Adversarial Network for geotechnicalsubsurface schematisation'],
        'Authors': ['Campos et al.'],
        'Year': [2025],
        'URL' : ['https://example.com/sample'],
        'Abstract': ['This is a sample abstract for demonstration purposes.'],
        'Journal': ['Journal of Geotechnical AI']
    }
    publications_df = pd.DataFrame(publications_data)
    
    # 2. Set directories
    pdf_dir = '.'  # Directory containing PDF files
    output_dir = './reports'  # Directory to save reports
    
    # 3. Initialize analyzer with your preferred Ollama model
    model_name = "qwq"  # Choose from models available in your Ollama installation
    analyzer = AIActComplianceAnalyzer(model_name=model_name, ollama_base_url="http://localhost:11434")
    
    # 4. Analyze publications and generate reports
    results_df = analyzer.batch_analyze_publications_with_direct_paths(publications_df, output_dir)
    
    # 5. Save summary results to CSV
    results_df.to_csv('compliance_summary.csv', index=False)
    
    print(f"Analysis complete. Reports saved to {output_dir}")
    print(f"Summary saved to compliance_summary.csv")