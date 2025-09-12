import React, { useState, useEffect, useRef } from 'react';
import { createRoot } from 'react-dom/client';

const App = () => {
  // File contents stored in state
  const [drawioContent, setDrawioContent] = useState(localStorage.getItem('drawioFileContent') || '');
  const [csvContent, setCsvContent] = useState(localStorage.getItem('csvFileContent') || '');
  const [ontologyIris, setOntologyIris] = useState(localStorage.getItem('ontologyIris') || 
`rico: https://www.ica.org/standards/RiC/ontology#
data: https://data.archives.gov.on.test.gbad.ca/
auth: https://data.archives.gov.on.test.gbad.ca/Schema/Authority/
add: https://data.archives.gov.on.test.gbad.ca/Schema/Description-Listings/
maps: https://data.archives.gov.on.test.gbad.ca/Schema/Mapping#
rdfs: http://www.w3.org/2000/01/rdf-schema#`);

  const [savedMessage, setSavedMessage] = useState('');
  const [output, setOutput] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');

  const pyodide = useRef<any>(null);

  // Show messages if files were already loaded from previous session
  useEffect(() => {
    if (drawioContent) setSavedMessage('Draw.io file already loaded from previous session.');
    if (csvContent) setSavedMessage(prev => prev 
      ? prev + ' CSV file already loaded from previous session.'
      : 'CSV file already loaded from previous session.');
  }, []);

  // Load Pyodide once
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

  // Handle conversion
  const handleConvert = async () => {
    if (!drawioContent || !csvContent || !pyodide.current) {
      setError('Please upload both a Draw.io and a CSV file.');
      return;
    }

    setIsLoading(true);
    setError('');
    setOutput('');

    try {
      // Load Python scripts
      const drawIOParserPy = await (await fetch('https://raw.githubusercontent.com/gbad-project/records_in_contexts_draw_io_parser/refs/heads/review/feat/pyodide-converter/pyodide-react/draw_io_parser.py')).text();
      const mapSchemaPy = await (await fetch('https://raw.githubusercontent.com/gbad-project/records_in_contexts_draw_io_parser/refs/heads/review/feat/pyodide-converter/pyodide-react/map_schema.py')).text();
      const preprocessorsPy = await (await fetch('https://raw.githubusercontent.com/gbad-project/records_in_contexts_draw_io_parser/refs/heads/review/feat/pyodide-converter/gbad/converter/preprocessors.py')).text();

      pyodide.current.FS.writeFile('/draw_io_parser.py', drawIOParserPy);
      pyodide.current.FS.writeFile('/map_schema.py', mapSchemaPy);

      // Write user files
      pyodide.current.FS.writeFile('/user.drawio', drawioContent);
      pyodide.current.FS.writeFile('/user.csv', csvContent);

      pyodide.current.globals.set('drawio_path', '/user.drawio');
      pyodide.current.globals.set('csv_path', '/user.csv');
      pyodide.current.globals.set('ontology_iris', ontologyIris);

      pyodide.current.FS.mkdir("/gbad");
      pyodide.current.FS.mkdir("/gbad/converter");
      pyodide.current.FS.writeFile("/gbad/converter/preprocessors.py", preprocessorsPy);


      const pythonCode = `
import sys
sys.path.append('/')
import map_schema
import draw_io_parser

#drawio_path = pyodide.globals.get('drawio_path')
#csv_path = pyodide.globals.get('csv_path')
#ontology_iris = pyodide.globals.get('ontology_iris')

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
      console.error(e);
      setError(String(e));
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
          <input
            type="file"
            accept=".drawio"
            onChange={async (e) => {
              const file = e.target.files?.[0];
              if (!file) return;
              const text = await file.text();
              setDrawioContent(text);
              localStorage.setItem('drawioFileContent', text);
              setSavedMessage(`Draw.io file "${file.name}" saved!`);
            }}
          />
        </label>
      </div>

      <div style={{ marginTop: '10px' }}>
        <label>
          CSV File:
          <input
            type="file"
            accept=".csv"
            onChange={async (e) => {
              const file = e.target.files?.[0];
              if (!file) return;
              const text = await file.text();
              setCsvContent(text);
              localStorage.setItem('csvFileContent', text);
              setSavedMessage(`CSV file "${file.name}" saved!`);
            }}
          />
        </label>
      </div>

      <div style={{ marginTop: '10px' }}>
        <label>
          Ontology IRIs:
          <textarea
            rows={6}
            style={{ width: '100%', verticalAlign: 'top' }}
            value={ontologyIris}
            onChange={(e) => {
              setOntologyIris(e.target.value);
              localStorage.setItem('ontologyIris', e.target.value);
            }}
          />
        </label>
      </div>

      <button onClick={handleConvert} disabled={isLoading} style={{ marginTop: '10px' }}>
        {isLoading ? 'Loading...' : 'Convert'}
      </button>

      {savedMessage && <div style={{ color: 'green', marginTop: '10px' }}>{savedMessage}</div>}
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

// render directly here
createRoot(document.getElementById("root")!).render(<App />);
