import csv
import json

def analyze_papers_against_checklist(papers_file, checklist_file, output_file):
    """
    Analyze papers against the checklist and generate a compliance report.

    Args:
        papers_file (str): The CSV file containing paper metadata.
        checklist_file (str): The text file containing the checklist.
        output_file (str): The JSON file to save the compliance report.
    """
    # Load checklist
    with open(checklist_file, "r", encoding="utf-8") as f:
        checklist = f.read()

    # Load papers
    papers = []
    with open(papers_file, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            papers.append(row)

    # Analyze each paper
    report = []
    for paper in papers:
        analysis = {
            "Title": paper["Title"],
            "Authors": paper["Authors"],
            "Year": paper["Year"],
            "Journal": paper["Journal"],
            "Checklist Compliance": {}
        }

        # Example: Check if the paper mentions "risk management" or "data governance"
        analysis["Checklist Compliance"]["Risk Management"] = "risk management" in paper["Title"].lower()
        analysis["Checklist Compliance"]["Data Governance"] = "data governance" in paper["Title"].lower()

        report.append(analysis)

    # Save report
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"Analysis complete. Report saved to {output_file}.")

if __name__ == "__main__":
    papers_file = "papers.csv"
    checklist_file = "checklist.txt"
    output_file = "compliance_report.json"

    analyze_papers_against_checklist(papers_file, checklist_file, output_file)
