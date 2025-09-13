import React, { useState, useEffect } from "react";

declare global {
  interface Window {
    loadPyodide: any;
    pyodideReady?: boolean;
    networkErrors?: string[];
  }
}

const App: React.FC = () => {
  const [fileContent, setFileContent] = useState<string>("");
  const [processedContent, setProcessedContent] = useState<string>("");
  const [pyodide, setPyodide] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string>("");
  const [pyodideStatus, setPyodideStatus] = useState<string>("Initializing…");

  useEffect(() => {
    const loadPyodideAsync = async () => {
      try {
        setPyodideStatus("Loading Pyodide from local files…");
        if (typeof window.loadPyodide === "undefined") {
          throw new Error("Pyodide script not loaded. Check local files.");
        }
        console.log("Starting Pyodide load...");
        const py = await window.loadPyodide({
          indexURL: "/pyodide/",
        });
        console.log("Pyodide loaded successfully");
        setPyodide(py);
        setPyodideStatus("Pyodide ready");
        window.pyodideReady = true;
        setLoading(false);
      } catch (err) {
        console.error("Failed to load Pyodide:", err);
        setError(
          `Failed to load Pyodide: ${err instanceof Error ? err.message : String(err)}`
        );
        setPyodideStatus("Failed to load Pyodide");
        setLoading(false);
      }
    };
    loadPyodideAsync();
  }, []);

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) {
      setError("No file selected");
      return;
    }
    if (!pyodide) {
      setError("Pyodide not ready yet. Please wait.");
      return;
    }
    try {
      setError("");
      setPyodideStatus("Processing file...");
      const text = await file.text();
      console.log("File content:", text);
      setFileContent(text);
      localStorage.setItem("uploadedFile", text);
      console.log("Fetching Python script...");
      const response = await fetch("script.py");
      if (!response.ok) {
        throw new Error(`Failed to fetch script.py: ${response.status} ${response.statusText}`);
      }
      const pythonCode = await response.text();
      console.log("Python script:", pythonCode);
      pyodide.globals.set("content", text);
      console.log("Running Python code...");
      const result = pyodide.runPython(pythonCode);
      console.log("Python result:", result);
      setProcessedContent(result as string);
      setPyodideStatus("File processed successfully");
    } catch (err) {
      console.error("Error processing file:", err);
      setError(`Error processing file: ${err instanceof Error ? err.message : String(err)}`);
      setPyodideStatus("Error during processing");
    }
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
      <div style={{ marginBottom: "1rem" }}>
        <strong>Status:</strong> {pyodideStatus}
      </div>
      {loading && (
        <div style={{ color: "blue" }}>Loading Pyodide... This may take a minute.</div>
      )}
      {error && (
        <div style={{ color: "red", marginBottom: "1rem" }}>
          <strong>Error:</strong> {error}
        </div>
      )}
      <input
        type="file"
        accept=".txt"
        onChange={handleFileChange}
        disabled={loading || !!error}
      />
      {fileContent && (
        <div style={{ marginTop: "1rem" }}>
          <h2>Original Content:</h2>
          <p style={{ background: "#f0f0f0", padding: "10px" }}>{fileContent}</p>
        </div>
      )}
      {processedContent && (
        <div style={{ marginTop: "1rem" }}>
          <h2>Processed Content:</h2>
          <p style={{ background: "#e8f5e8", padding: "10px" }}>{processedContent}</p>
          <button onClick={downloadFile}>Download Processed File</button>
        </div>
      )}
      <div style={{ marginTop: "2rem", fontSize: "0.8em", color: "#666" }}>
        <details>
          <summary>Debug Info</summary>
          <div>Pyodide loaded: {pyodide ? "Yes" : "No"}</div>
          <div>Loading: {loading ? "Yes" : "No"}</div>
          <div>Error: {error || "None"}</div>
          <div>File content length: {fileContent.length}</div>
          <div>Processed content: {processedContent || "None"}</div>
        </details>
      </div>
    </div>
  );
};

export default App;
