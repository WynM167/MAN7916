from pathlib import Path

training_folder = Path("local_data") / "topic_model" / "train"
pdf_files = sorted(training_folder.glob("*.pdf"))

print(f"Found {len(pdf_files)} training PDFs")
for pdf_path in pdf_files:
    print(pdf_path.name)
# checks whether Python can find the ten PDFs

