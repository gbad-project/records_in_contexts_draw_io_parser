import { test, expect, afterAll } from "bun:test";
import { server } from "./app";

const drawioFilePath = "gbad/schema/description-listings/General ADD (Descriptions and Listings) to RiC-O Model_2025-06-20_PZ.drawio";
const csvFilePath = "gbad/mapping/source/generic.csv";
const expectedRmlPath = "pyodide/expected.rml";
const ontologyIris = `rico: https://www.ica.org/standards/RiC/ontology#
data: https://data.archives.gov.on.test.gbad.ca/
auth: https://data.archives.gov.on.test.gbad.ca/Schema/Authority/
add: https://data.archives.gov.on.test.gbad.ca/Schema/Description-Listings/
maps: https://data.archives.gov.on.test.gbad.ca/Schema/Mapping#`;


afterAll(() => {
  server.stop();
});

test("POST /convert with drawio and csv files returns RML", async () => {
  const drawioFile = Bun.file(drawioFilePath);
  const csvFile = Bun.file(csvFilePath);
  const expectedRml = await Bun.file(expectedRmlPath).text();

  const formData = new FormData();
  formData.append("drawioFile", new File([await drawioFile.arrayBuffer()], drawioFile.name, { type: drawioFile.type }));
  formData.append("csvFile", new File([await csvFile.arrayBuffer()], csvFile.name, { type: csvFile.type }));
  formData.append("ontologyIris", ontologyIris);

  const response = await server.fetch(
    new Request("http://localhost:3000/convert", {
      method: "POST",
      body: formData,
    })
  );

  expect(response.status).toBe(200);

  const responseText = await response.text();

  // Normalize both expected and actual RML content
  const normalize = (str: string) => str.replace(/\s+/g, ' ').trim();

  const normalizedExpected = normalize(expectedRml);
  const normalizedResponse = normalize(responseText);

  expect(normalizedResponse).toBe(normalizedExpected);
}, 20000); // 20 seconds timeout for pyodide loading and processing
