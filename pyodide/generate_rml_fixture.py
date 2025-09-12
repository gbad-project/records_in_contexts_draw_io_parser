import os
import sys
from pathlib import Path

# Add root directory to path to allow imports of map_schema and draw_io_parser
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import draw_io_parser
import map_schema

# Define file paths
drawio_file = "gbad/schema/description-listings/General ADD (Descriptions and Listings) to RiC-O Model_2025-06-20_PZ.drawio"
csv_file = "generic.csv"
temp_ttl_file = "pyodide/temp_schema.ttl"
output_rml_file = "pyodide/expected.rml"

print("Generating schema graph from draw.io file...")
# 1. Parse the drawio file to a graph
schema_graph = draw_io_parser.parse_drawio_to_graph(
    drawio_file,
    metacharacter_substitute=['url', ' =%20'],
    capitalisation_scheme='none',
    ontology_iri='https://data.archives.gov.on.test.gbad.ca/Schema/Mapping',
    include_label=False
)

print(f"Schema graph generated with {len(schema_graph)} triples.")

# 2. Save the graph to a temporary TTL file
print(f"Saving schema graph to {temp_ttl_file}...")
schema_graph.serialize(destination=temp_ttl_file, format='turtle')
print("Schema graph saved.")

# 3. Generate RML from the TTL file and CSV file
print(f"Generating RML from {temp_ttl_file} and {csv_file}...")
rml_content = map_schema.__init__('generic', csv_file, graph_path=temp_ttl_file)
print("RML content generated.")

# 4. Save the RML content to the output file
print(f"Saving RML content to {output_rml_file}...")
with open(output_rml_file, "w") as f:
    f.write(rml_content)

print("RML fixture generated successfully.")

# Clean up the temporary file
os.remove(temp_ttl_file)
print(f"Removed temporary file {temp_ttl_file}.")
