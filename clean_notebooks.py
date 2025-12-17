import os
import json
import glob

def clean_notebook(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Check if 'widgets' exists in metadata and remove it
    if 'metadata' in data and 'widgets' in data['metadata']:
        print(f"🧹 Cleaning widgets metadata from: {file_path}")
        del data['metadata']['widgets']
        
        # Save the clean version back
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)
            # Note: indent=1 mimics Colab's default saving style to minimize diffs
    else:
        print(f"✅ No widgets found in: {file_path}")

# Find all .ipynb files recursively
notebooks = glob.glob('**/*.ipynb', recursive=True)

print(f"Found {len(notebooks)} notebooks. Checking for invalid metadata...")
for nb in notebooks:
    clean_notebook(nb)

print("\n✨ Done! You can now commit and push the fixed notebooks.")