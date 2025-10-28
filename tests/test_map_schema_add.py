import unittest
import os
import shutil
from rdflib import Graph, Namespace, RDF, BNode
from map_schema import __init__ as map_schema_init
from draw_io_parser import parse_drawio_to_graph
import glob

# Define namespaces
rr = Namespace("http://www.w3.org/ns/r2rml#")
rml = Namespace("http://semweb.mmlab.be/ns/rml#")

def _get_node_properties(graph, subject):
    """Recursively gets properties of a node, handling blank nodes."""
    if isinstance(subject, BNode):
        properties = []
        for p, o in sorted(graph.predicate_objects(subject=subject)):
            properties.append((p, _get_node_properties(graph, o)))
        return frozenset(properties)
    else:
        return subject

def get_triples_map_representation(graph, triples_map_subject):
    """Returns a canonical, hashable representation of a TriplesMap."""
    logical_source = graph.value(subject=triples_map_subject, predicate=rml.logicalSource)
    subject_map = graph.value(subject=triples_map_subject, predicate=rr.subjectMap)
    predicate_object_maps = graph.objects(subject=triples_map_subject, predicate=rr.predicateObjectMap)

    raw_logical_source_props = _get_node_properties(graph, logical_source)
    # Filter out the rml:source predicate, which can differ between test runs
    logical_source_props = frozenset(prop for prop in raw_logical_source_props if prop[0] != rml.source)

    subject_map_props = _get_node_properties(graph, subject_map)

    pom_props_set = set()
    for pom in predicate_object_maps:
        pom_props_set.add(_get_node_properties(graph, pom))

    return (logical_source_props, subject_map_props, frozenset(pom_props_set))

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

        graph_dir = 'gbad/schema/description-listings'
        source_csv_filename = "temp/test_description_tailshuf_100.csv"
        ttl_path = glob.glob(os.path.join(graph_dir, "*.ttl"))[0]
        graph_name = os.path.splitext(os.path.basename(ttl_path))[0]
        generated_rml_dir = f'{graph_dir}/{os.path.splitext(source_csv_filename)[0]}'
        if os.path.exists(generated_rml_dir):
            shutil.rmtree(generated_rml_dir)

    def test_map_schema_add_extended_comparison(self):
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

        # The generated rml path is constructed inside map_schema.py
        graph_dir = 'gbad/schema/description-listings'
        ttl_path = glob.glob(os.path.join(graph_dir, "*.ttl"))[0]
        graph_name = os.path.splitext(os.path.basename(ttl_path))[0]
        generated_rml_path = f'{graph_dir}/{os.path.splitext(source_csv_filename)[0]}/{graph_name}.rml'

        # Load graphs
        ground_truth_graph = Graph().parse(ground_truth_rml_path, format="turtle")
        generated_graph = Graph().parse(generated_rml_path, format="turtle")

        # Get all TriplesMaps from both graphs
        gt_triples_maps = set(ground_truth_graph.subjects(predicate=RDF.type, object=rr.TriplesMap))
        gen_triples_maps = set(generated_graph.subjects(predicate=RDF.type, object=rr.TriplesMap))

        # Get canonical representations for each TriplesMap
        gt_representations = {get_triples_map_representation(ground_truth_graph, tm) for tm in gt_triples_maps}
        gen_representations = {get_triples_map_representation(generated_graph, tm) for tm in gen_triples_maps}

        # Compare the sets of representations
        self.assertEqual(gt_representations, gen_representations)

if __name__ == '__main__':
    unittest.main()
