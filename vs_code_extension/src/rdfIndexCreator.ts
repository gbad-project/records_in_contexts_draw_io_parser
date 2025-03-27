import { Parser, Quad } from 'n3';
import * as fs from "fs";

export class RDFIndexCreator {
    private index: Set<string>;

    constructor() {
        this.index = new Set();
    }

    addTriples(ntriples: string): void {
        const parser = new Parser();
        parser.parse(ntriples, (error: Error | null, quad: Quad | null) => {
            if (error) throw error;
            if (quad) {
                this.index.add(quad.subject.value);
                this.index.add(quad.object.value);
            }
        });
    }

    has(uri: string): boolean {
        return this.index.has(uri);
    }

    // **Step 1: Dump Index to JSON File**
    dumpToFile(filename: string): void {
        fs.writeFileSync(filename, JSON.stringify([...this.index]), "utf8");
        console.log(`Index dumped to ${filename}`);
    }
}
