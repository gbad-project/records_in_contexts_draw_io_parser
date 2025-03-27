import * as N3 from 'n3';

export class RDFProcessor {
    private parser: N3.Parser;
    private store: N3.Store;

    constructor() {
        this.parser = new N3.Parser();
        this.store = new N3.Store();
    }

    /**
     * Validate a Turtle/RDF file
     * @param fileContent File content to validate
     * @returns Validation result
     */
    public validate(fileContent: string): { valid: boolean; errors?: string[] } {
        const errors: string[] = [];

        try {
            // Clear previous store
            this.store.removeQuads(this.store.getQuads(null, null, null, null));

            // Parse and store quads
            this.parser.parse(fileContent, (error: Error | null, quad: N3.Quad | null) => {
                if (error) {
                    errors.push(error.message);
                } else if (quad) {
                    this.store.addQuad(quad);
                }
            });

            return {
                valid: errors.length === 0,
                errors: errors.length > 0 ? errors : undefined
            };
        } catch (err) {
            return {
                valid: false,
                errors: [err instanceof Error ? err.message : String(err)]
            };
        }
    }

    /**
     * Serialize the current store to specified format
     * @param format Serialization format
     * @returns Serialized RDF
     */
    public serialize(format: string = 'turtle'): string {
        const writer = this.createWriter(format);
        return writer.quadsToString(this.store.getQuads(null, null, null, null));
    }

    /**
     * Create a writer based on the specified format
     * @param format Serialization format
     * @returns N3 Writer
     */
    private createWriter(format: string): N3.Writer {
        switch (format.toLowerCase()) {
            case 'n3':
                return new N3.Writer({ format: 'N3' });
            case 'nquads':
                return new N3.Writer({ format: 'N-Quads' });
            case 'jsonld':
                return new N3.Writer({ format: 'JSON-LD' });
            case 'turtle':
            default:
                return new N3.Writer({ format: 'Turtle' });
        }
    }
}
