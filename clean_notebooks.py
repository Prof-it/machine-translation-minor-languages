"""
Utility to clean Jupyter notebooks for GitHub rendering.
Removes ipywidgets metadata that breaks GitHub previews.
"""

import json
import glob

def clean_notebook(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cleaned = False

    # Remove problematic widget metadata
    if 'metadata' in data and 'widgets' in data['metadata']:
        print(f"🧹 Cleaning widgets metadata from: {file_path}")
        del data['metadata']['widgets']
        cleaned = True

        # Remove empty metadata if needed
        if not data['metadata']:
            del data['metadata']

    if cleaned:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1)
    else:
        print(f"✅ No widgets found in: {file_path}")

# Find all notebooks recursively
notebooks = glob.glob('**/*.ipynb', recursive=True)

print(f"Found {len(notebooks)} notebooks. Checking for invalid metadata...")
for nb in notebooks:
    clean_notebook(nb)

print("\n✨ Done! Notebooks are GitHub-safe.")
