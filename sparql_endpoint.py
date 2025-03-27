from rdflib import Dataset
from rdflib_endpoint import SparqlEndpoint
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables from file
load_dotenv(".env.local")
GBAD_ROOT = os.getenv("GBAD_ROOT")
GRAPH_PATH = os.getenv("GRAPH_PATH")
LOCALHOST = os.getenv("LOCALHOST")
PORT = int(os.getenv("PORT"))

# Create a Dataset and configure the SPARQL endpoint
g = Dataset()

# Replace this with your RDF triples or graph
g.parse(os.path.join(GBAD_ROOT, GRAPH_PATH), format="nt")

app = SparqlEndpoint(
    graph=g,
    path="/sparql",  # This is the SPARQL endpoint URL
    #service_description="text/turtle", # Unsupported by rdflib
    cors_enabled=True,
    title="SPARQL endpoint for RDFLib graph",
    description="A SPARQL endpoint for RDFLib graph",
    version="0.1.0",
    example_query="""SELECT (COUNT(*) AS ?count)
WHERE {
  ?s ?p ?o .
}""",
)

# Start the SPARQL endpoint with a custom port (3030)
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="debug")
