import unittest
import os
import shutil
from rdflib import Graph, Namespace, RDF
from map_schema import __init__ as map_schema_init
from draw_io_parser import parse_drawio_to_graph
import glob

class TestMapSchemaAdd(unittest.TestCase):
    def setUp(self):
        self.tests_temp_dir = "tests/temp"
        os.makedirs(self.tests_temp_dir, exist_ok=True)

        self.gbad_temp_dir = "gbad/mapping/source/temp"
        os.makedirs(self.gbad_temp_dir, exist_ok=True)

        source_csv_path = "gbad/mapping/source/tests/test_description_tailshuf_100.csv"
        temp_csv_path = os.path.join(self.gbad_temp_dir, "test_description_tailshuf_100.csv")
        shutil.copy(source_csv_path, temp_csv_path)

    def tearDown(self):
        shutil.rmtree(self.tests_temp_dir)
        shutil.rmtree(self.gbad_temp_dir)

        # also remove the generated rml file's directory
        graph_dir = 'gbad/schema/description-listings'
        source_csv_filename = "temp/test_description_tailshuf_100.csv"
        ttl_path = glob.glob(os.path.join(graph_dir, "*.ttl"))[0]
        graph_name = os.path.splitext(os.path.basename(ttl_path))[0]
        generated_rml_dir = f'{graph_dir}/{os.path.splitext(source_csv_filename)[0]}'
        if os.path.exists(generated_rml_dir):
            shutil.rmtree(generated_rml_dir)


    def test_map_schema_add(self):
        # Define file paths
        ground_truth_rml_path = "gbad/schema/description-listings/tests/test_description_tailshuf_100/general_add_descriptions_and_listings_to_ric-o_model_2025-06-20_pz.rml"
        drawio_path = "gbad/schema/description-listings/General ADD (Descriptions and Listings) to RiC-O Model_2025-06-20_PZ.drawio"
        source_csv_filename = "temp/test_description_tailshuf_100.csv"

        # Generate TTL from draw.io
        generated_ttl_path = os.path.join(self.tests_temp_dir, "generated.ttl")
        g = parse_drawio_to_graph(
            drawio_path,
            metacharacter_substitute=['url', ' =%20'],
            capitalisation_scheme='none',
            ontology_iri='https://data.archives.gov.on.test.gbad.ca/Schema/Mapping',
            include_label=False
        )
        g.serialize(destination=generated_ttl_path, format='turtle')

        # Generate RML
        map_schema_init('add', source_csv_filename)

        # The generated rml path is constructed inside map_schema.py, so I need to reconstruct it here
        graph_dir = 'gbad/schema/description-listings'
        ttl_path = glob.glob(os.path.join(graph_dir, "*.ttl"))[0]
        graph_name = os.path.splitext(os.path.basename(ttl_path))[0]
        generated_rml_path = f'{graph_dir}/{os.path.splitext(source_csv_filename)[0]}/{graph_name}.rml'

        # Load graphs
        ground_truth_graph = Graph()
        ground_truth_graph.parse(ground_truth_rml_path, format="turtle")

        generated_graph = Graph()
        generated_graph.parse(generated_rml_path, format="turtle")

        # Define namespaces
        rr = Namespace("http://www.w3.org/ns/r2rml#")

        # Count TriplesMaps
        ground_truth_triples_maps = list(ground_truth_graph.subjects(predicate=RDF.type, object=rr.TriplesMap))
        generated_triples_maps = list(generated_graph.subjects(predicate=RDF.type, object=rr.TriplesMap))

        # Count PredicateObjectMaps
        ground_truth_po_maps = list(ground_truth_graph.objects(subject=None, predicate=rr.predicateObjectMap))
        generated_po_maps = list(generated_graph.objects(subject=None, predicate=rr.predicateObjectMap))

        # Assert counts are equal
        self.assertEqual(len(ground_truth_triples_maps), len(generated_triples_maps))
        self.assertEqual(len(ground_truth_po_maps), len(generated_po_maps))

if __name__ == '__main__':
    unittest.main()
