"""
Tests for the DrawioParser class.
"""

import unittest
from pathlib import Path
from argparse import Namespace

from draw_io_parser import DrawioParser, _arguments_parser

class TestDrawioParserClass(unittest.TestCase):
    """
    Tests the DrawioParser class.
    """

    def test_parser_with_file(self):
        """
        Tests that the DrawioParser class can parse a draw.io file and return
        the expected OWL output.
        """
        drawio_file_path = "gbad/schema/description-listings/General ADD (Descriptions and Listings) to RiC-O Model_2025-06-20_PZ.drawio"
        with open(drawio_file_path, "r", encoding="utf-8") as f:
            drawio_content = f.read()

        parser = DrawioParser()
        args = _arguments_parser().parse_args(['-m', 'url'])

        # Set up default arguments for testing
        args.prefix = ["rico", "owl", "rdfs"]
        args.prefix_iri = ["https://www.ica.org/standards/RiC/ontology#", "http://www.w3.org/2002/07/owl#", "http://www.w3.org/2000/01/rdf-schema#"]

        output = parser.run(drawio_content, args)
        self.assertIsNotNone(output)
        self.assertIsInstance(output, str)
        self.assertGreater(len(output), 0)
