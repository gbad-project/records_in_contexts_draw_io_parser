import { loadPyodide } from "./node_modules/pyodide/pyodide.mjs";
import { readdir, stat } from "fs/promises";
import { join } from "path";

async function initializePyodide() {
  const pyodide = await loadPyodide({
    indexURL: "./node_modules/pyodide/build",
  });
  await pyodide.loadPackage("micropip");
  const micropip = pyodide.pyimport("micropip");
  await micropip.install("pandas");
  await micropip.install("rdflib");
  await micropip.install("requests");
  await micropip.install("lxml");
  return pyodide;
}

const pyodidePromise = initializePyodide();

async function copyToPyodideFS(pyodide: any, source: string, destination: string) {
  await pyodide.FS.mkdirTree(destination);
  const files = await readdir(source);
  for (const file of files) {
    const sourcePath = join(source, file);
    const destPath = join(destination, file);
    const stats = await stat(sourcePath);
    if (stats.isDirectory()) {
      await copyToPyodideFS(pyodide, sourcePath, destPath);
    } else {
      const content = await Bun.file(sourcePath).arrayBuffer();
      pyodide.FS.writeFile(destPath, new Uint8Array(content));
    }
  }
}

export const server = Bun.serve({
  port: 3000,
  async fetch(req) {
    const url = new URL(req.url);

    if (url.pathname === "/") {
      return new Response(Bun.file("pyodide/index.html"));
    }

    if (url.pathname === "/convert" && req.method === "POST") {
      const formData = await req.formData();
      const drawioFile = formData.get("drawioFile") as File;
      const ontologyIris = formData.get("ontologyIris") as string;
      const csvFile = formData.get("csvFile") as File;
      const preProcessorFile = formData.get("preProcessorFile") as File | null;
      const postProcessorFile = formData.get("postProcessorFile") as File | null;

      const pyodide = await pyodidePromise;

      const drawioContent = await drawioFile.text();
      const csvContent = await csvFile.text();
      const preProcessorContent = preProcessorFile ? await preProcessorFile.text() : "";
      const postProcessorContent = postProcessorFile ? await postProcessorFile.text() : "";

      // Write user files to predictable locations in Pyodide FS
      pyodide.FS.writeFile("/user_drawio.drawio", drawioContent);
      pyodide.FS.writeFile("/user_csv.csv", csvContent);
      if (preProcessorContent) {
        pyodide.FS.writeFile("/user_preprocessor.py", preProcessorContent);
      }
      if (postProcessorContent) {
        pyodide.FS.writeFile("/user_postprocessor.py", postProcessorContent);
      }

      // Load repo files into Pyodide FS
      const mapSchemaPy = await Bun.file("/app/map_schema.py").text();
      const drawIOParserPy = await Bun.file("/app/draw_io_parser.py").text();
      pyodide.FS.writeFile("/map_schema.py", mapSchemaPy);
      pyodide.FS.writeFile("/draw_io_parser.py", drawIOParserPy);
      await copyToPyodideFS(pyodide, "/app/gbad", "/gbad");

      const prefixes = {};
      const ontologyIriLines = ontologyIris.split('\n');
      for (const line of ontologyIriLines) {
          const parts = line.split(':');
          if (parts.length >= 2) {
              const prefix = parts[0].trim();
              const iri = parts.slice(1).join(':').trim();
              prefixes[prefix] = iri;
          }
      }

      pyodide.globals.set("drawio_content", drawioContent);
      pyodide.globals.set("prefixes", prefixes);

      const pythonWrapper = `
import os
import sys
import importlib.util

# Set up the environment to mimic running from the repo root
sys.path.append("/")
os.chdir("/")

import draw_io_parser
import map_schema
prefixes = prefixes.to_py()

rml_content = ""
try:
    # 1. Parse the user's drawio file content
    schema_graph = draw_io_parser.parse_drawio_content_to_graph(
        drawio_content,
        custom_prefixes=prefixes,
        metacharacter_substitute=[' =_']
    )

    # 2. Save the graph to a temporary TTL file
    schema_ttl_path = "/temp_schema.ttl"
    schema_graph.serialize(destination=schema_ttl_path, format='turtle')

    # 3. Handle pre-processor
    csv_path = "/user_csv.csv"
    if os.path.exists("/user_preprocessor.py"):
        spec = importlib.util.spec_from_file_location("user_preprocessor", "/user_preprocessor.py")
        user_preprocessor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(user_preprocessor)

        preprocessed_csv_path = "/preprocessed_user_csv.csv"
        user_preprocessor.preprocess(csv_path, preprocessed_csv_path)
        csv_path = preprocessed_csv_path

    # 4. Generate RML
    rml_content = map_schema.__init__('generic', csv_path, graph_path=schema_ttl_path)

    # 5. Handle post-processor
    if os.path.exists("/user_postprocessor.py"):
        spec = importlib.util.spec_from_file_location("user_postprocessor", "/user_postprocessor.py")
        user_postprocessor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(user_postprocessor)
        rml_content = user_postprocessor.postprocess(rml_content)

except Exception as e:
    import traceback
    rml_content = traceback.format_exc()

rml_content
      `;

      const rmlResult = await pyodide.runPythonAsync(pythonWrapper);

      return new Response(rmlResult, {
        headers: {
            "Content-Disposition": "attachment; filename=output.rml",
            "Content-Type": "application/rdf+xml",
        }
      });
    }

    return new Response("Not Found", { status: 404 });
  },
});

console.log(`Listening on http://localhost:${server.port}`);
