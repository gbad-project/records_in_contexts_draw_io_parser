import React, { useState, useEffect, useRef } from 'react';
import { createRoot } from 'react-dom/client';

const App = () => {
  const [drawioFile, setDrawioFile] = useState<File | null>(null);
  const [csvFile, setCsvFile] = useState<File | null>(null);
  const [ontologyIris, setOntologyIris] = useState(
`rico: https://www.ica.org/standards/RiC/ontology#
data: https://data.archives.gov.on.test.gbad.ca/
auth: https://data.archives.gov.on.test.gbad.ca/Schema/Authority/
add: https://data.archives.gov.on.test.gbad.ca/Schema/Description-Listings/
maps: https://data.archives.gov.on.test.gbad.ca/Schema/Mapping#
rdfs: http://www.w3.org/2000/01/rdf-schema#`);
  const [output, setOutput] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');

  const pyodide = useRef<any>(null);

  useEffect(() => {
    const loadPyodide = async () => {
      try {
        // @ts-ignore
        pyodide.current = await window.loadPyodide({
          indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.28.2/full/',
        });
        await pyodide.current.loadPackage(['micropip']);
        const micropip = pyodide.current.pyimport('micropip');
        await micropip.install('pandas');
        await micropip.install('rdflib');
        await micropip.install('requests');
        await micropip.install('lxml');
        setIsLoading(false);
      } catch (e) {
        setError('Failed to load Pyodide.');
        setIsLoading(false);
      }
    };
    loadPyodide();
  }, []);

  const handleConvert = async () => {
    if (!drawioFile || !csvFile || !pyodide.current) {
      setError('Please select both a .drawio and a .csv file.');
      return;
    }

    setIsLoading(true);
    setError('');
    setOutput('');

    try {
      const drawioContent = await drawioFile.text();
      const csvContent = await csvFile.text();

      // Load Python scripts
      const drawIOParserPy = await (await fetch('./draw_io_parser.py')).text();
      const mapSchemaPy = await (await fetch('./map_schema.py')).text();

      pyodide.current.FS.writeFile('/draw_io_parser.py', drawIOParserPy);
      pyodide.current.FS.writeFile('/map_schema.py', mapSchemaPy);

      // Write user files
      pyodide.current.FS.writeFile('/user.drawio', drawioContent);
      pyodide.current.FS.writeFile('/user.csv', csvContent);

      pyodide.current.globals.set('drawio_path', '/user.drawio');
      pyodide.current.globals.set('csv_path', '/user.csv');
      pyodide.current.globals.set('ontology_iris', ontologyIris);

      const pythonCode = `
import sys
sys.path.append('/')
import map_schema
import draw_io_parser

drawio_path = pyodide.globals.get('drawio_path')
csv_path = pyodide.globals.get('csv_path')
ontology_iris = pyodide.globals.get('ontology_iris')

prefixes = {}
for line in ontology_iris.split('\\n'):
    parts = line.split(':')
    if len(parts) >= 2:
        prefix = parts[0].strip()
        iri = ':'.join(parts[1:]).strip()
        prefixes[prefix] = iri

with open(drawio_path, 'r') as f:
    drawio_content = f.read()

try:
    schema_graph = draw_io_parser.parse_drawio_content_to_graph(
        drawio_content,
        custom_prefixes=prefixes,
        metacharacter_substitute=[' =_']
    )

    rml_content = map_schema.__init__('generic', csv_path, graph_path=schema_graph)
    rml_content
except Exception as e:
    import traceback
    traceback.format_exc()
`;

      const result = await pyodide.current.runPythonAsync(pythonCode);
      setOutput(result);
    } catch (e) {
      setError('An error occurred during the conversion process.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={{ padding: '20px' }}>
      <h1>Draw.io to RML Converter</h1>
      <div>
        <label>
          Draw.io File:
          <input type="file" accept=".drawio" onChange={(e) => setDrawioFile(e.target.files ? e.target.files[0] : null)} />
        </label>
      </div>
      <div style={{ marginTop: '10px' }}>
        <label>
          CSV File:
          <input type="file" accept=".csv" onChange={(e) => setCsvFile(e.target.files ? e.target.files[0] : null)} />
        </label>
      </div>
      <div style={{ marginTop: '10px' }}>
        <label>
          Ontology IRIs:
          <textarea
            rows={6}
            style={{ width: '100%', verticalAlign: 'top' }}
            value={ontologyIris}
            onChange={(e) => setOntologyIris(e.target.value)}
          />
        </label>
      </div>
      <button onClick={handleConvert} disabled={isLoading} style={{ marginTop: '10px' }}>
        {isLoading ? 'Loading...' : 'Convert'}
      </button>
      {error && <div style={{ color: 'red', marginTop: '10px' }}>{error}</div>}
      {output && (
        <div style={{ marginTop: '20px' }}>
          <h2>Output:</h2>
          <pre style={{ border: '1px solid #ccc', padding: '10px', whiteSpace: 'pre-wrap', wordWrap: 'break-word' }}>
            {output}
          </pre>
        </div>
      )}
    </div>
  );
};

export default App;
