import * as fs from 'fs';
import path from 'path';
import drawIoParserPy from './draw_io_parser.py?raw';

async function main() {
  // Fake a browser-like environment for Pyodide.
  // @ts-ignore
  delete (globalThis as any).process;
  // @ts-ignore
  (globalThis as any).window = globalThis;
  // @ts-ignore
  (globalThis as any).self = globalThis;
  // @ts-ignore
  (globalThis as any).document = {};
  // @ts-ignore
  (globalThis as any).location = { href: '' };
  // @ts-ignore
  (globalThis as any).navigator = {};
  const { loadPyodide } = await import('pyodide');
  const pyodide = await loadPyodide({
    indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.28.2/full/',
  });
  await pyodide.loadPackage(['rdflib', 'lxml']);

  pyodide.FS.writeFile('/draw_io_parser.py', drawIoParserPy);

  const drawioContent = fs.readFileSync(path.join(__dirname, '../General Authority to RiC-O Model_2024-11-25_PZ.drawio'), 'utf-8');
  pyodide.FS.writeFile('/user.drawio', drawioContent);

  const ontologyIris = `rico: https://www.ica.org/standards/RiC/ontology#\n`
    + `data: https://data.archives.gov.on.test.gbad.ca/\n`
    + `auth: https://data.archives.gov.on.test.gbad.ca/Schema/Authority/\n`
    + `add: https://data.archives.gov.on.test.gbad.ca/Schema/Description-Listings/\n`
    + `maps: https://data.archives.gov.on.test.gbad.ca/Schema/Mapping#\n`
    + `rdfs: http://www.w3.org/2000/01/rdf-schema#`;

  pyodide.globals.set('drawio_path', '/user.drawio');
  pyodide.globals.set('ontology_iris', ontologyIris);

  const pythonCode = `
import sys
sys.path.append('/')
import draw_io_parser

prefixes = {}
for line in ontology_iris.split('\n'):
    parts = line.split(':')
    if len(parts) >= 2:
        prefix = parts[0].strip()
        iri = ':'.join(parts[1:]).strip()
        prefixes[prefix] = iri

with open(drawio_path, 'r') as f:
    drawio_content = f.read()

print('drawio content length', len(drawio_content))

schema_graph = draw_io_parser.parse_drawio_content_to_graph(
    drawio_content,
    custom_prefixes=prefixes,
    metacharacter_substitute=[' =_']
)

print('parsed graph with', len(schema_graph), 'triples')
print(schema_graph.serialize(format='turtle'))
`;

  const result = await pyodide.runPythonAsync(pythonCode);
  console.log(result);
}

main();
