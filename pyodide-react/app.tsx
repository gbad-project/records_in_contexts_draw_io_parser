import React, { useState, useEffect, useRef } from 'react';
import { createRoot } from 'react-dom/client';

const App = () => {
  // File contents stored in state
  const [drawioContent, setDrawioContent] = useState(
    localStorage.getItem('drawioFileContent') || ''
  );

  const [savedMessage, setSavedMessage] = useState('');
  const [output, setOutput] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');

  const pyodide = useRef<any>(null);

  // Show messages if files were already loaded from previous session
  useEffect(() => {
    if (drawioContent)
      setSavedMessage('Draw.io file already loaded from previous session.');
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
        await micropip.install('rdflib');
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
    if (!drawioContent || !pyodide.current) {
      setError('Please upload a Draw.io file.');
      return;
    }

    setIsLoading(true);
    setError('');
    setOutput('');

    try {
      // Load Python scripts
      const parserPy = await (
        await fetch('./draw_io_parser_simple.py')
      ).text();

      pyodide.current.FS.writeFile('/draw_io_parser_simple.py', parserPy);
      pyodide.current.FS.writeFile('/user.drawio', drawioContent);

      const pythonCode = `
import sys
sys.path.append('/')
import draw_io_parser_simple as parser

with open('/user.drawio', 'r') as f:
    drawio_content = f.read()

parser.parse_drawio_content_to_graph(drawio_content)
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
