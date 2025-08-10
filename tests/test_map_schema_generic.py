import unittest
import os
import sys
from rdflib import Graph, Namespace
from rdflib.namespace import RDF

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import map_schema

class TestMapSchemaGeneric(unittest.TestCase):

    def test_map_schema_generic(self):
        # Run the map_schema script
        map_schema.__init__('generic', 'generic.csv')

        # Check that the output file was created
        rml_path = 'gbad/schema/generic/generic/generic_schema.rml'
        self.assertTrue(os.path.exists(rml_path))

        # Load the generated RML file
        g = Graph()
        g.parse(rml_path, format="turtle")

        # Check the number of TriplesMaps
        rr = Namespace("http://www.w3.org/ns/r2rml#")
        triples_maps = list(g.subjects(RDF.type, rr.TriplesMap))
        self.assertEqual(len(triples_maps), 2)

if __name__ == '__main__':
    unittest.main()
