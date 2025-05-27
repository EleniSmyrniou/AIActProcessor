import os
import json
import pandas as pd
import PyPDF2
import requests
from tqdm import tqdm
import numpy as np
from ollama_api import OllamaAPI  # Import the OllamaAPI class
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class RAGVectorStore:
    """Vector store for RAG implementation"""
    
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        """
        Initialize the vector store
        
        Args:
            embedding_model: Name of the sentence-transformers model to use
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.chunks = []
        self.embeddings = None
        self.metadata = []
    
    def add_chunks(self, chunks, chunk_metadata=None):
        """
        Add text chunks to the vector store and generate embeddings
        
        Args:
            chunks: List of text chunks
            chunk_metadata: List of metadata dicts for each chunk (optional)
        """
        if not chunks:
            return
            
        # Store chunks and metadata
        start_idx = len(self.chunks)
        self.chunks.extend(chunks)
        
        # Generate embeddings for new chunks
        new_embeddings = self.embedding_model.encode(chunks)
        
        # Add metadata for each chunk
        if chunk_metadata:
            self.metadata.extend(chunk_metadata)
        else:
            self.metadata.extend([{} for _ in chunks])
            
        # Update embeddings array
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
            
        return list(range(start_idx, start_idx + len(chunks)))
    
    def search(self, query, top_k=5):
        """
        Search for most relevant chunks based on a query
        
        Args:
            query: Query text
            top_k: Number of top results to return
            
        Returns:
            List of (chunk_idx, score, text, metadata) tuples
        """
        if not self.chunks or self.embeddings is None:
            return []
            
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Calculate cosine similarity
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get indices of top-k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Return results
        results = []
        for idx in top_indices:
            results.append({
                "chunk_idx": idx,
                "score": float(similarities[idx]),
                "text": self.chunks[idx],
                "metadata": self.metadata[idx]
            })
            
        return results


class AIActComplianceAnalyzer:
    def __init__(self, ollama_base_url="http://localhost:11434", model_name="response_qwen3", 
                embedding_model="all-MiniLM-L6-v2", chunk_size=1000, chunk_overlap=200):
        """
        Initialize the AI Act Compliance Analyzer with RAG capabilities
        
        Args:
            ollama_base_url: URL where Ollama is running
            model_name: Name of the Ollama model to use (e.g., llama3, mistral, etc.)
            embedding_model: Name of the sentence-transformers model to use
            chunk_size: Size of text chunks for document splitting
            chunk_overlap: Overlap between consecutive chunks
        """
        self.ollama_api = OllamaAPI(url=ollama_base_url, model=model_name)
        self.vector_store = RAGVectorStore(embedding_model=embedding_model)
        self.checklist = self._load_ai_act_checklist()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
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
    
    def chunk_text(self, text, page_numbers=None):
        """
        Split text into chunks with overlap
        
        Args:
            text: Text to split
            page_numbers: List of page numbers for each character (optional)
            
        Returns:
            List of text chunks and their metadata
        """
        chunks = []
        chunk_metadata = []
        
        # Split by paragraphs first
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = ""
        current_length = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # If adding this paragraph exceeds chunk size and we already have content,
            # save the current chunk and start a new one with overlap
            para_length = len(para)
            if current_length + para_length > self.chunk_size and current_length > 0:
                chunks.append(current_chunk)
                
                # Create metadata for this chunk
                chunk_metadata.append({
                    "chunk_id": len(chunks) - 1,
                })
                
                # Start new chunk with overlap
                overlap_size = min(self.chunk_overlap, current_length)
                current_chunk = current_chunk[-overlap_size:] + "\n" + para
                current_length = overlap_size + 1 + para_length
            else:
                if current_length > 0:
                    current_chunk += "\n"
                    current_length += 1
                current_chunk += para
                current_length += para_length
                
            # If current chunk is already at chunk size, save it
            if current_length >= self.chunk_size:
                chunks.append(current_chunk)
                chunk_metadata.append({
                    "chunk_id": len(chunks) - 1,
                })
                current_chunk = ""
                current_length = 0
        
        # Add the last chunk if there's content left
        if current_length > 0:
            chunks.append(current_chunk)
            chunk_metadata.append({
                "chunk_id": len(chunks) - 1,
            })
        
        return chunks, chunk_metadata
    
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
    
    def analyze_compliance_with_rag(self, pdf_path, publication_metadata=None):
        """
        Analyze a publication's compliance with the AI Act checklist using RAG
        
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
        
        # Chunk the text and add to vector store
        print("Chunking document and generating embeddings...")
        chunks, chunk_metadata = self.chunk_text(pdf_text)
        
        # Add page/section info to metadata if available
        if publication_metadata:
            for metadata in chunk_metadata:
                counter = 0
                metadata.update({
                    "title": publication_metadata.get('title', 'Unknown'),
                    "authors": publication_metadata.get('authors', 'Unknown'),
                    "year": publication_metadata.get('year', 'Unknown'),
                    "page_number": counter
                })
                counter += 1
        
        # Add chunks to vector store
        self.vector_store.add_chunks(chunks, chunk_metadata)
        
        # Prepare metadata string
        metadata_str = ""
        if publication_metadata:
            metadata_str = f"Title: {publication_metadata.get('title', 'Unknown')}\n"
            metadata_str += f"Authors: {publication_metadata.get('authors', 'Unknown')}\n"
            metadata_str += f"Year: {publication_metadata.get('year', 'Unknown')}\n\n"
        
        results = {}
        
        # Analyze each category separately
        for category, items in tqdm(self.checklist.items(), desc="Analyzing categories"):
            category_results = {
                "category": category,
                "items": [],
                "average_score": 0,
                "summary": ""
            }
            
            # Process each item in the checklist for this category
            item_scores = []
            # use tqdm to show progress
            for i, item in tqdm(enumerate(items), total=len(items), desc=f"Analyzing {category} items"):
                # Construct search query for this specific checklist item
                search_query = f"{category}: {item}"
                
                # Retrieve relevant chunks from the vector store
                relevant_chunks = self.vector_store.search(search_query, top_k=3)
                
                # Extract the text from relevant chunks
                context_texts = [result['text'] for result in relevant_chunks]
                
                # Construct a combined context with the most relevant information
                context = "\n\n---\n\n".join(context_texts)
                
                # Create the analysis prompt for this specific checklist item
                prompt = f"""
                You are an expert in AI ethics and regulation. Your task is to determine if a research publication
                in geotechnics AI addresses the following AI Act requirement:
                
                Category: {category}
                Requirement: {item}
                
                {metadata_str}
                
                I have retrieved the most relevant sections from the document for you to analyze:
                
                {context}
                
                Rate how well this requirement is addressed on a scale of 0-5:
                0: Not mentioned at all
                1: Briefly mentioned but no details
                2: Some details but inadequate
                3: Moderately addressed
                4: Well addressed
                5: Comprehensively addressed with best practices
                
                Provide your analysis in JSON format with the following structure:
                {{
                  "item": "{item}",
                  "score": X,  // number between 0-5
                  "justification": "Brief explanation of why you assigned this score, with specific references to the text"
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
                        json_match = re.search(r'\{.*\}', json_str, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(0)
                        else:
                            raise ValueError("No JSON found in response")
                    
                    item_result = json.loads(json_str)
                    category_results["items"].append(item_result)
                    item_scores.append(item_result.get("score", 0))
                except Exception as e:
                    print(f"Error processing item '{item}': {e}")
                    # Add error item
                    category_results["items"].append({
                        "item": item,
                        "score": 0,
                        "justification": f"Error analyzing this item: {str(e)}"
                    })
                    item_scores.append(0)
            
            # Calculate average score for this category
            if item_scores:
                category_results["average_score"] = sum(item_scores) / len(item_scores)
            
            # Generate a summary for this category
            summary_prompt = f"""
            You are an expert in AI ethics and regulation. Based on the following scores for the category "{category}",
            provide a brief summary (2-3 sentences) of the document's compliance with this aspect of the AI Act:
            
            {json.dumps(category_results["items"], indent=2)}
            
            Average score: {category_results["average_score"]}/5.0
            
            Provide ONLY the summary text with no additional formatting.
            """
            
            summary_response = self.query_ollama(summary_prompt)
            category_results["summary"] = summary_response.strip()
            
            results[category] = category_results
        
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

                # Reset vector store for each new document
                self.vector_store = RAGVectorStore()
                
                # Analyze compliance using RAG
                results = self.analyze_compliance_with_rag(pdf_path, metadata)

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
                print(f"Error analyzing publication {row.get('Document', '')}: {e}")
                # Add error summary
                summary = {
                    'filename': os.path.basename(row.get('Document', '')),
                    'title': row.get('Title', ''),
                    'authors': row.get('Authors', ''),
                    'year': row.get('Year', ''),
                    'report_path': '',
                    'overall_score': 0,
                    'error': str(e)
                }
                results_summary.append(summary)
        
        return pd.DataFrame(results_summary)
