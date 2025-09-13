from xml.etree.ElementTree import fromstring
from rdflib import Graph, URIRef, RDF, Namespace

EX = Namespace("http://example.org/")

def parse_drawio_content_to_graph(drawio_content: str) -> str:
    """Parse minimal .drawio XML and return a Turtle string.

    This simplified parser only creates one triple for each vertex found in the
    diagram. The triple asserts that the vertex is an ``ex:Node``.
    """
    tree = fromstring(drawio_content)
    graph = Graph()
    graph.bind("ex", EX)
    for cell in tree.findall(".//mxCell[@vertex='1']"):
        node_id = cell.get("id")
        if node_id:
            graph.add((URIRef(EX[node_id]), RDF.type, EX.Node))
    return graph.serialize(format="turtle")
