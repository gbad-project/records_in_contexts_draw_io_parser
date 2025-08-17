import os
from pathlib import Path
from unittest import TestCase
from rdflib import Graph, Namespace, RDF
import map_schema
from draw_io_parser import parse_drawio_to_graph
import tempfile
import shutil

class TestAuthoritySchemaMapping(TestCase):
    def test_end_to_end_map_schema_auth(self):
        self.maxDiff = None

        # Define files and directories
        project_root = Path(__file__).resolve().parent.parent
        drawio_file = project_root / "gbad/schema/authority/General Authority to RiC-O Model_2025-06-25_PZ.drawio"
        source_csv_filename = "test_authority_tailshuf_100.csv"
        source_csv = project_root / "gbad/mapping/source/tests" / source_csv_filename
        ground_truth_rml_file = project_root / "gbad/schema/authority/tests/test_authority_tailshuf_100/general_authority_to_ric-o_model_2025-06-25_pz.rml"

        # Create a temporary directory to store generated files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)

            # Create the necessary directory structure inside the temp directory
            temp_gbad_schema_auth_dir = temp_dir_path / "gbad/schema/authority"
            temp_gbad_schema_auth_dir.mkdir(parents=True, exist_ok=True)

            # Copy the source csv to the temp directory structure
            temp_source_dir = temp_dir_path / "gbad/mapping/source"
            temp_source_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(source_csv, temp_source_dir / source_csv_filename)

            # Generate the TTL file from the draw.io file and save it in the temp dir
            generated_ttl_file = temp_gbad_schema_auth_dir / "general_authority_to_ric-o_model_2025-06-25_pz.ttl"
            os.environ['BASE_URI'] = 'https://data.archives.gov.on.test.gbad.ca'
            generated_graph = parse_drawio_to_graph(
                drawio_file,
                metacharacter_substitute=['url', ' =%20'],
                capitalisation_scheme='none',
                ontology_iri='https://data.archives.gov.on.test.gbad.ca/Schema/Mapping',
                include_label=False
            )
            generated_graph.serialize(destination=str(generated_ttl_file), format="turtle")

            # Temporarily change the working directory to the temp directory
            original_cwd = Path.cwd()
            os.chdir(temp_dir_path)

            try:
                # Generate the RML file
                map_schema.__init__('auth', source_csv_filename)

                # Construct the path to the generated RML file
                graph_name = os.path.splitext(os.path.basename(str(generated_ttl_file)))[0]
                generated_rml_file = f'gbad/schema/authority/{os.path.splitext(source_csv_filename)[0]}/{graph_name}.rml'

                # Load the generated RML graph
                generated_rml_graph = Graph()
                generated_rml_graph.parse(generated_rml_file, format="turtle")

                # Load the ground truth RML graph
                ground_truth_rml_graph = Graph()
                ground_truth_rml_graph.parse(str(ground_truth_rml_file), format="turtle")

                # Define namespaces
                rr = Namespace("http://www.w3.org/ns/r2rml#")

                # Count rr:TriplesMap in both graphs
                generated_triples_maps_query = "SELECT (COUNT(*) as ?count) WHERE { ?s a rr:TriplesMap . }"
                generated_triples_maps_result = generated_rml_graph.query(generated_triples_maps_query, initNs={'rr': rr})
                generated_triples_maps = int(list(generated_triples_maps_result)[0]['count'])

                ground_truth_triples_maps_query = "SELECT (COUNT(*) as ?count) WHERE { ?s a rr:TriplesMap . }"
                ground_truth_triples_maps_result = ground_truth_rml_graph.query(ground_truth_triples_maps_query, initNs={'rr': rr})
                ground_truth_triples_maps = int(list(ground_truth_triples_maps_result)[0]['count'])

                # Count rr:predicateObjectMap in both graphs
                generated_pom_query = "SELECT (COUNT(*) as ?count) WHERE { ?s rr:predicateObjectMap ?o . }"
                generated_pom_result = generated_rml_graph.query(generated_pom_query, initNs={'rr': rr})
                generated_predicate_object_maps = int(list(generated_pom_result)[0]['count'])

                ground_truth_pom_query = "SELECT (COUNT(*) as ?count) WHERE { ?s rr:predicateObjectMap ?o . }"
                ground_truth_pom_result = ground_truth_rml_graph.query(ground_truth_pom_query, initNs={'rr': rr})
                ground_truth_predicate_object_maps = int(list(ground_truth_pom_result)[0]['count'])

                # Assertions
                self.assertEqual(generated_triples_maps, ground_truth_triples_maps, "Number of rr:TriplesMap should be the same.")
                self.assertEqual(generated_predicate_object_maps, ground_truth_predicate_object_maps, "Number of rr:predicateObjectMap should be the same.")

            finally:
                # Change back to the original working directory
                os.chdir(original_cwd)
