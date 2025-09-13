import React, { useState, useEffect } from "react";

declare global {
  interface Window {
    loadPyodide: any;
  }
}

const App: React.FC = () => {
  const [fileContent, setFileContent] = useState<string>("");
  const [processedContent, setProcessedContent] = useState<string>("");
  const [pyodide, setPyodide] = useState<any>(null);

  useEffect(() => {
    const loadPyodideAsync = async () => {
      const py = await window.loadPyodide({
        indexURL: "https://cdn.jsdelivr.net/pyodide/v0.23.4/full/",
      });
      setPyodide(py);
      (window as any).pyodideReady = true;
    };
    loadPyodideAsync();
  }, []);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !pyodide) return;
    const text = await file.text();
    setFileContent(text);
    localStorage.setItem("uploadedFile", text);

    const pythonCode = await (await fetch("script.py")).text();
    pyodide.globals.set("content", text);
    const result = pyodide.runPython(pythonCode);
    setProcessedContent(result as string);
  };

  const downloadFile = () => {
    const blob = new Blob([processedContent], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "processed_file.txt";
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div style={{ padding: "2rem" }}>
      <h1>File Upload + Pyodide Processing</h1>
      <input type="file" accept=".txt" onChange={handleFileChange} />
      {processedContent && (
        <div>
          <h2>Processed Content:</h2>
          <p>{processedContent}</p>
          <button onClick={downloadFile}>Download Processed File</button>
        </div>
      )}
    </div>
  );
};

export default App;
