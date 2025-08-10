"""
Tests the DrawIOXMLTree class in the way it would be used when running
draw_io_parser.py
"""

import os
from pathlib import Path
from unittest import TestCase
from rdflib import Graph
from rdflib.compare import isomorphic, graph_diff

from draw_io_parser import parse_drawio_to_graph


class TestDrawIOParser(TestCase):
    """
    Tests that the parsing of the .drawio files in the examples/ directory
    gives the expected results
    """

    def test_end_to_end(self):
        """
        Tests that parsing the specified draw.io file produces a graph
        isomorphic to the ground truth turtle file.
        """
        self.maxDiff = None
        drawio_file = "gbad/schema/description-listings/General ADD (Descriptions and Listings) to RiC-O Model_2025-06-20_PZ.drawio"
        ttl_file = "gbad/schema/description-listings/general_add_descriptions_and_listings_to_ric-o_model_2025-06-20_pz.ttl"

        # Set the base URI for the test
        os.environ['BASE_URI'] = 'https://data.archives.gov.on.test.gbad.ca'


        # Generate the graph from the draw.io file
        generated_graph = parse_drawio_to_graph(
            drawio_file,
            metacharacter_substitute=['url', ' =%20'],
            capitalisation_scheme='none',
            ontology_iri='https://data.archives.gov.on.test.gbad.ca/Schema/Mapping',
            include_label=False
        )

        # Load the ground truth graph
        ground_truth_graph = Graph()
        ground_truth_graph.parse(ttl_file, format="turtle")

        # Compare the graphs
        are_isomorphic = isomorphic(generated_graph, ground_truth_graph)
        if not are_isomorphic:
            in_both, in_generated, in_ground_truth = graph_diff(generated_graph, ground_truth_graph)
            print("--- In Generated Graph Only ---")
            for t in in_generated:
                print(t)
            print("--- In Ground Truth Graph Only ---")
            for t in in_ground_truth:
                print(t)
        self.assertTrue(are_isomorphic, "The generated graph is not isomorphic to the ground truth graph.")

    def test_end_to_end_authority(self):
        """
        Tests the second graph for isomorphism.
        """
        self.maxDiff = None
        project_root = Path(__file__).resolve().parent.parent
        drawio_file = project_root / "gbad/schema/authority/General Authority to RiC-O Model_2025-06-25_PZ.drawio"
        ttl_file = project_root / "gbad/schema/authority/general_authority_to_ric-o_model_2025-06-25_pz.ttl"

        # Set the base URI for the test
        os.environ['BASE_URI'] = 'https://data.archives.gov.on.test.gbad.ca'

        # Generate the graph from the draw.io file
        generated_graph = parse_drawio_to_graph(
            drawio_file,
            metacharacter_substitute=['url', ' =%20'],
            capitalisation_scheme='none',
            ontology_iri='https://data.archives.gov.on.test.gbad.ca/Schema/Mapping',
            include_label=False
        )

        # Load the ground truth graph
        ground_truth_graph = Graph()
        ground_truth_graph.parse(ttl_file, format="turtle")

        # Compare the graphs
        are_isomorphic = isomorphic(generated_graph, ground_truth_graph)
        if not are_isomorphic:
            in_both, in_generated, in_ground_truth = graph_diff(generated_graph, ground_truth_graph)
            print("--- In Generated Graph Only (Authority Test) ---")
            for t in in_generated:
                print(t)
            print("--- In Ground Truth Graph Only (Authority Test) ---")
            for t in in_ground_truth:
                print(t)
        self.assertTrue(are_isomorphic, "The generated authority graph is not isomorphic to the ground truth graph.")
