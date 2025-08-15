import unittest
import os
import sys
from rdflib import Graph
from rdflib.compare import isomorphic, graph_diff

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import map_schema
import draw_io_parser

class TestMapSchemaGeneric(unittest.TestCase):

    def setUp(self):
        """Set up test environment."""
        self.base_dir = "gbad/schema/"
        os.environ['BASE_URI'] = 'https://data.archives.gov.on.test.gbad.ca'
        #This is needed to avoid FileNotFoundError
        if not os.path.exists("gbad/mapping/source/preprocessed"):
            os.makedirs("gbad/mapping/source/preprocessed")
        if not os.path.exists("gbad/schema/authority/authority_tailshuf_100"):
            os.makedirs("gbad/schema/authority/authority_tailshuf_100")
        if not os.path.exists("gbad/schema/description-listings/description_tailshuf_100"):
            os.makedirs("gbad/schema/description-listings/description_tailshuf_100")
        if not os.path.exists("gbad/schema/generic/generic"):
            os.makedirs("gbad/schema/generic/generic")


    def _run_map_schema_and_compare(self, schema_code, source_ttl_path, ground_truth_rml_path, source_csv_filename=None):
        """Helper function to run map_schema and compare the output with a ground truth file."""

        print(f"[{schema_code}] Running map_schema...")
        # Determine the generated rml path from ttl path
        graph_name = os.path.splitext(os.path.basename(source_ttl_path))[0]
        if schema_code == 'add':
            schema_folder = 'description-listings'
            source_filename_no_ext = os.path.splitext(source_csv_filename)[0]
            generated_rml_path = f"{self.base_dir}{schema_folder}/{source_filename_no_ext}/{graph_name}.rml"
        elif schema_code == 'auth':
            schema_folder = 'authority'
            source_filename_no_ext = os.path.splitext(source_csv_filename)[0]
            generated_rml_path = f"{self.base_dir}{schema_folder}/{source_filename_no_ext}/{graph_name}.rml"
        elif schema_code == 'generic':
            schema_folder = 'generic'
            source_filename_no_ext = os.path.splitext(source_csv_filename)[0]
            generated_rml_path = f"{self.base_dir}{schema_folder}/{source_filename_no_ext}/{graph_name}.rml"
        else:
            generated_rml_path = source_ttl_path.replace('.ttl', '.rml')

        # Run map_schema
        map_schema.__init__(schema_code, source_filename=source_csv_filename)
        print(f"[{schema_code}] map_schema finished.")

        # Check that the output file was created
        self.assertTrue(os.path.exists(generated_rml_path), f"File not created at {generated_rml_path}")

        # Load graphs
        print(f"[{schema_code}] Loading generated graph from {generated_rml_path}...")
        generated_graph = Graph()
        generated_graph.parse(generated_rml_path, format="turtle")
        print(f"[{schema_code}] Generated graph loaded. Size: {len(generated_graph)}")

        print(f"[{schema_code}] Loading ground truth graph from {ground_truth_rml_path}...")
        ground_truth_graph = Graph()
        ground_truth_graph.parse(ground_truth_rml_path, format="turtle")
        print(f"[{schema_code}] Ground truth graph loaded. Size: {len(ground_truth_graph)}")

        # Compare graphs
        print(f"[{schema_code}] Comparing graphs...")

        are_isomorphic = isomorphic(generated_graph, ground_truth_graph)

        if not are_isomorphic:
            in_both, in_generated, in_ground_truth = graph_diff(generated_graph, ground_truth_graph)
            print("--- In Generated Graph Only ---")
            for t in in_generated:
                print(t)
            print("--- In Ground Truth Graph Only ---")
            for t in in_ground_truth:
                print(t)

        self.assertTrue(are_isomorphic,
                        f"Graphs are not isomorphic for schema {schema_code}")
        print(f"[{schema_code}] Graphs are isomorphic.")

    def test_description_listings(self):
        """Test the description-listings schema."""
        source_ttl_path = f"{self.base_dir}description-listings/general_add_descriptions_and_listings_to_ric-o_model_2025-06-20_pz.ttl"
        ground_truth_rml_path = f"{self.base_dir}description-listings/tests/test_description_tailshuf_100/general_add_descriptions_and_listings_to_ric-o_model_2025-06-20_pz.rml"
        self._run_map_schema_and_compare("add", source_ttl_path, ground_truth_rml_path, source_csv_filename="description_tailshuf_100.csv")

    def test_authority(self):
        """Test the authority schema."""
        source_ttl_path = f"{self.base_dir}authority/general_authority_to_ric-o_model_2025-06-25_pz.ttl"
        ground_truth_rml_path = f"{self.base_dir}authority/tests/test_authority_tailshuf_100/general_authority_to_ric-o_model_2025-06-25_pz.rml"
        self._run_map_schema_and_compare("auth", source_ttl_path, ground_truth_rml_path, source_csv_filename="authority_tailshuf_100.csv")

    def test_generic(self):
        """Test the generic schema."""
        # Define paths
        drawio_path = f"{self.base_dir}generic/generic_schema.drawio"
        generic_ttl_path = f"{self.base_dir}generic/generic_schema.ttl"

        # Generate TTL from drawio
        print("[generic] Generating TTL from drawio...")
        generated_graph = draw_io_parser.parse_drawio_to_graph(
            drawio_path,
            metacharacter_substitute=['url', ' =%20'],
            capitalisation_scheme='none',
            ontology_iri='https://data.archives.gov.on.test.gbad.ca/Schema/Mapping',
            include_label=False
        )
        generated_graph.serialize(destination=generic_ttl_path, format="turtle")
        self.assertTrue(os.path.exists(generic_ttl_path))
        print("[generic] TTL generated.")

        # Generate ground truth with 'auth'
        print("[generic] Generating ground truth with 'auth'...")
        map_schema.__init__("auth", source_filename="generic.csv")
        auth_rml_path = f"{self.base_dir}generic/generic/generic_schema.rml"
        self.assertTrue(os.path.exists(auth_rml_path))
        print("[generic] Ground truth with 'auth' generated.")

        # Generate ground truth with 'add'
        print("[generic] Generating ground truth with 'add'...")
        map_schema.__init__("add", source_filename="generic.csv")
        add_rml_path = f"{self.base_dir}generic/generic/generic_schema.rml"
        self.assertTrue(os.path.exists(add_rml_path))
        print("[generic] Ground truth with 'add' generated.")

        # Compare 'auth' and 'add' generated graphs
        print("[generic] Comparing 'auth' and 'add' graphs...")
        auth_graph = Graph().parse(auth_rml_path, format="turtle")
        add_graph = Graph().parse(add_rml_path, format="turtle")
        self.assertTrue(isomorphic(auth_graph, add_graph), "Auth and Add generic graphs are not isomorphic")
        print("[generic] 'Auth' and 'add' graphs are isomorphic.")

        # Run the generic test
        print("[generic] Running generic test...")
        self._run_map_schema_and_compare("generic", generic_ttl_path, auth_rml_path, source_csv_filename="generic.csv")
        print("[generic] Generic test finished.")


if __name__ == '__main__':
    # To focus on a single test, comment out the others
    # unittest.main()

    # Or run a specific test like this:
    suite = unittest.TestSuite()
    suite.addTest(TestMapSchemaGeneric('test_authority'))
    runner = unittest.TextTestRunner()
    runner.run(suite)
