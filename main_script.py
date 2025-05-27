import os
import pandas as pd
from pdf_analyzer import AIActComplianceAnalyzer
#from rag_pdf_analyzer import AIActComplianceAnalyzer as RAGAnalyzer

def load_publications_database(csv_path):
    """
    Load publications database from CSV file
    
    Args:
        csv_path: Path to CSV file with publications data
        
    Returns:
        pd.DataFrame: DataFrame with publications data
    """
    try:
        df = pd.read_csv(csv_path)
        expected_columns = ['Title', 'Authors', 'Year', 'Journal', 'URL', 'Abstract', 'Document']
        
        # Check if expected columns exist
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            print(f"Warning: Missing columns in database: {missing_columns}")
            # Add missing columns with empty values
            for col in missing_columns:
                df[col] = ""
        
        # Create a filename column from the Document path
        df['filename'] = df['Document'].apply(lambda x: os.path.basename(x) if isinstance(x, str) else "")
        
        # Map columns to expected format
        df['title'] = df['Title']
        df['authors'] = df['Authors']
        df['year'] = df['Year']
        
        return df
    except Exception as e:
        print(f"Error loading publications database: {e}")
        # Create empty DataFrame with required columns
        return pd.DataFrame(columns=['filename', 'title', 'authors', 'year', 'Title', 'Authors', 'Year', 'Journal', 'URL', 'Abstract', 'Document'])

def main():
    # Configuration
    pdf_dir = "."  # Directory containing PDF files (can be overridden if using full paths)
    output_dir = "./reports_2"  # Directory to save reports
    database_path = "./geotechnical_ai_papers.csv"  # Path to CSV file with publications data
    model_name = "qwq"  # Choose from models available in your Ollama installation
    use_direct_paths = True  # Set to True to use Document paths directly instead of pdf_dir
    
    # Create directories if they don't exist
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    print("AI Act Compliance Analyzer")
    print("=========================")
    
    # Check if Ollama is running
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            available_models = [model["name"] for model in response.json().get("models", [])]
            print(f"Available Ollama models: {', '.join(available_models)}")
            
            if model_name not in available_models:
                print(f"Warning: Selected model '{model_name}' not found. Available models: {', '.join(available_models)}")
        else:
            print("Warning: Could not retrieve available models from Ollama.")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to Ollama. Make sure it's running on localhost:11434.")
        print("You can start Ollama with: 'ollama serve'")
        return
    
    # Load publications database
    print(f"Loading publications database from {database_path}")
    if os.path.exists(database_path):
        publications_df = load_publications_database(database_path)
        print(f"Loaded {len(publications_df)} publications")
    else:
        print(f"Database file not found: {database_path}")
        print("Creating a sample database for demonstration")
        
        # Create sample data
        publications_data = {
            'Title': ['Sample AI in Geotechnics Paper'],
            'Authors': ['Sample Author'],
            'Year': [2023],
            'Journal': ['Journal of Geotechnical AI'],
            'URL': ['https://example.com/sample'],
            'Abstract': ['This is a sample abstract for demonstration purposes.'],
            'Document': ['sample_publication.pdf']
        }
        publications_df = pd.DataFrame(publications_data)
        
        # Save sample database
        publications_df.to_csv(database_path, index=False)
        print(f"Created sample database at {database_path}")
        
        # Add derived columns
        publications_df['filename'] = publications_df['Document'].apply(lambda x: os.path.basename(x))
        publications_df['title'] = publications_df['Title']
        publications_df['authors'] = publications_df['Authors']
        publications_df['year'] = publications_df['Year']
    
    # Check for PDF files
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        print("Please add your PDF files to this directory.")
        return
    
    print(f"Found {len(pdf_files)} PDF files in {pdf_dir}")
    
    # Initialize analyzer
    print(f"Initializing analyzer with model: {model_name}")
    analyzer = AIActComplianceAnalyzer(model_name=model_name, ollama_base_url="http://localhost:11434",)
    #analyzer = RAGAnalyzer(model_name=model_name, 
    #                        ollama_base_url="http://localhost:11434",
    #                            chunk_size=2000,  # Size of text chunks (adjust as needed),
    #                            chunk_overlap=200,  # Overlap between chunks (adjust as needed)
    #                            embedding_model_name="all-MiniLM-L6-v2" )
    
    # Filter publications to only those with PDF files available
    available_files = [f for f in publications_df['filename'] if f in pdf_files]
    if not available_files:
        print("No publications in the database match the available PDF files.")
        print("Please ensure filenames in the database match the PDF files.")
        return
    
    filtered_df = publications_df[publications_df['filename'].isin(pdf_files)]
    print(f"Will analyze {len(filtered_df)} publications that match the database")
    
    # Analyze publications
    print("Starting analysis. This may take some time depending on the number and size of publications.")
    results_df = analyzer.batch_analyze_publications_with_direct_paths(filtered_df, output_dir)
    
    # Save results
    results_path = "ai_act_compliance_results.csv"
    results_df.to_csv(results_path, index=False)
    
    print("\nAnalysis complete!")
    print(f"- Individual reports saved to: {output_dir}")
    print(f"- Summary results saved to: {results_path}")
    
    # Print summary statistics
    if not results_df.empty:
        avg_score = results_df['overall_score'].mean()
        print(f"\nAverage compliance score across all publications: {avg_score:.2f}/5.0")
        
        # Category compliance
        categories = [col.replace('_score', '') for col in results_df.columns if col.endswith('_score') and col != 'overall_score']
        print("\nCompliance by category:")
        for category in categories:
            category_avg = results_df[f'{category}_score'].mean()
            print(f"- {category}: {category_avg:.2f}/5.0")

if __name__ == "__main__":
    main()