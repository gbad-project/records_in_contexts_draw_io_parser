# Originally generated with Claude 3.7 Sonnet on 2025-04-14, modified

#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from rdflib import Dataset
from rdflib import Dataset, URIRef, Namespace
import hashlib
import base64
import uuid

BASE_URI = 'https://data.archives.gov.on.test.gbad.ca/'

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
                    print(f"\nLoading {file_path}\n")
                    # Create a temporary dataset for this file
                    temp_dataset = Dataset()
                    temp_dataset.parse(str(file_path), format=format_map[ext])

                    # Create named graph URI for this file
                    def create_named_graph_uri_from_file(file_path: Path,
                                                         graph_uri_path: URIRef):
                        encoding_to_use = 'utf-8'
                        file_content = file_path.read_text(encoding=encoding_to_use)
                        file_hash = hashlib.sha256(file_content.encode(encoding_to_use))
                        print(f"SHA256 hash '{file_hash.hexdigest()}' generated for '{encoding_to_use}' encoded plain text contents of: '{file_path}'")
                        print(f"Base64: {base64.b64encode(file_hash.digest()).decode()}")
                        
                        # https://www.rfc-editor.org/rfc/rfc6920.html
                        # Figure 4: ni Name Syntax
                        # NI-URI         = ni-scheme ":" ni-hier-part [ "?" query ]
                        # ni-scheme      = "ni"
                        # ni-hier-part   = "//" [ authority ] "/" alg-val
                        # alg-val        = alg ";" val
                        # The "val" field MUST contain the output of base64url encoding
                        # (with no "=" padding characters)
                        # Note also that ni val must be generated from bytes and URL safe
                        ni_val = base64.urlsafe_b64encode(file_hash.digest()).decode().rstrip('=')
                        ni_uri = f"ni:///sha-256;{ni_val}"
                        print(f"Valid ni URI (RFC6920-compliant): <{ni_uri}>")

                        # Generate a UUID v5 for ni URI
                        uuid_str = uuid.uuid5(uuid.NAMESPACE_URL, ni_uri)
                        print(f"UUID v5 generated from ns:URL and <{ni_uri}>: {uuid_str}")
                        # Combine into graph
                        graph_uri = URIRef(os.path.join(graph_uri_path, f"urn:uuid:{uuid_str}"))
                        print(f"Graph URI to be used for '{os.path.basename(file_path)}' ({file_hash.hexdigest()[:8]}): <{graph_uri}>")
                        print(f"""Steps to verify (ONLY for public data!):
1. Copy and paste file contents into: https://emn178.github.io/online-tools/sha256.html
2. Make sure input encoding is set to UTF-8 and output encoding to Base64
3. Review the output - should be identical to Base64
4. Once the hash is verified, copy and paste entire ni URI here: https://www.uuidtools.com/v5
5. Make sure you select 'Enter identifier for pre-defined UUIDs' and choose 'ns:URL - for URLs'
6. The UUID should match that in the graph URI""")
                        return graph_uri
                    graph_uri = create_named_graph_uri_from_file(
                        file_path=file_path,
                        graph_uri_path=URIRef(f"{BASE_URI[:-1]}/graph/")
                    )

                    # Add triples to the named graph
                    for quad in temp_dataset.quads():
                        s, p, o, g = quad
                        if g is None or str(g) == "urn:x-rdflib:default":  # Default graph
                            dataset.add((s, p, o, graph_uri))
                        else:  # Already in a named graph
                            dataset.add((s, p, o, g))
                    
                    file_count += 1
                except Exception as e:
                    print(f"Error processing {file_path}: {e}", file=sys.stderr)
    
    # Write the consolidated dataset to the output file as N-Quads
    if file_count > 0:
        print(f"\nWriting {len(dataset)} quads to {output_file}")
        dataset.serialize(destination=output_file, format='nquads')
        print(f"Consolidated {file_count} files successfully\n")
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
