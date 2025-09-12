import { test, expect, describe } from "bun:test";

describe("Server", () => {
    test("should return index.html on GET /", async () => {
        const res = await fetch("http://localhost:3000/");
        expect(res.status).toBe(200);
        const text = await res.text();
        expect(text).toContain("<h1>DrawIO to RML Converter</h1>");
    });

    test("should return 404 on unknown route", async () => {
        const res = await fetch("http://localhost:3000/unknown");
        expect(res.status).toBe(404);
    });

    // This is a basic test for the /convert endpoint.
    // A more comprehensive test would require mocking the pyodide environment
    // and the python scripts, which is complex.
    test("should handle POST /convert", async () => {
        const formData = new FormData();
        formData.append("drawioFile", new File(["<drawio></drawio>"], "test.drawio", { type: "application/xml" }));
        formData.append("csvFile", new File(["a,b,c"], "test.csv", { type: "text/csv" }));
        formData.append("ontologyIris", "ex: http://example.com/");

        const res = await fetch("http://localhost:3000/convert", {
            method: "POST",
            body: formData,
        });

        expect(res.status).toBe(200);
        expect(res.headers.get("Content-Disposition")).toBe("attachment; filename=output.rml");
        expect(res.headers.get("Content-Type")).toBe("application/rdf+xml");
    });
});
