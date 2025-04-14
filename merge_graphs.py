# Originally generated with Claude 3.7 Sonnet on 2025-04-14, modified

#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from rdflib import Dataset

def consolidate_rdf_files(input_dir, output_file):
    """
    Read all supported RDF files in a directory and combine them into a single N-Quads file.
    
    Args:
        input_dir: Directory containing RDF files
        output_file: Path to output N-Quads file
    """
    # Create a single dataset to hold all triples/quads
    dataset = Dataset()
    
    # File extensions and their corresponding formats
    format_map = {
        '.nt': 'nt',           # N-Triples
        '.nq': 'nquads',       # N-Quads
        '.ttl': 'turtle',      # Turtle
        '.trig': 'trig',       # TriG
        '.rdf': 'xml',         # RDF/XML
        '.owl': 'xml',         # OWL (XML format)
        '.jsonld': 'json-ld',  # JSON-LD
        '.n3': 'n3'            # Notation3
    }
    
    # Count loaded files
    file_count = 0
    
    # Process each file in the directory
    for file_path in Path(input_dir).glob('*'):
        if file_path.is_file():
            ext = file_path.suffix.lower()
            if ext in format_map:
                try:
                    print(f"Loading {file_path}")
                    dataset.parse(str(file_path), format=format_map[ext])
                    file_count += 1
                except Exception as e:
                    print(f"Error processing {file_path}: {e}", file=sys.stderr)
    
    # Write the consolidated dataset to the output file as N-Quads
    if file_count > 0:
        print(f"Writing {len(dataset)} quads to {output_file}")
        dataset.serialize(destination=output_file, format='nquads')
        print(f"Consolidated {file_count} files successfully")
    else:
        print("No valid RDF files found")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_directory> <output_file>")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_file = sys.argv[2]
    
    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a directory", file=sys.stderr)
        sys.exit(1)
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    consolidate_rdf_files(input_dir, output_file)
