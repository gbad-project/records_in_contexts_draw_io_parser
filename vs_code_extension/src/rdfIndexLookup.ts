import { Parser, Quad } from "n3";
import * as fs from "fs";
import fetch from "node-fetch";

export class RDFIndexLookup {
    private index: Set<string>;

    constructor(index: Set<string>) {
        this.index = index;
    }

    checkGraph(ntriples: string): string {
        const parser = new Parser();
        let conflicts: string[] = [];

        parser.parse(ntriples, (error: Error | null, quad?: Quad) => {
            if (error) throw error;
            if (quad) {
                if (this.index.has(quad.subject.value)) {
                    conflicts.push(`Conflict: Subject ${quad.subject.value} found in index`);
                }
                if (this.index.has(quad.object.value)) {
                    conflicts.push(`Conflict: Object ${quad.object.value} found in index`);
                }
            }
        });

        if (conflicts.length > 0) {
            console.error("Errors detected:");
            conflicts.forEach(c => console.error(c));
            throw new Error("Graph contains indexed subjects/objects!");
        }

        return "Success: No conflicts found!";
    }

    // **Step 2: Load Index from Local File**
    loadFromFile(filename: string): void {
        const data: string[] = JSON.parse(fs.readFileSync(filename, "utf8"));
        this.index = new Set(data);
        console.log(`Index loaded from ${filename}`);
    }

    // **Step 3: Load Index from External URL**
    async loadFromUrl(url: string): Promise<void> {
        console.log(`Fetching index from ${url}...`);
        const response = await fetch(url);
        const data: string[] = await response.json();
        this.index = new Set(data);
        console.log(`Index loaded from ${url}`);
    }
}
