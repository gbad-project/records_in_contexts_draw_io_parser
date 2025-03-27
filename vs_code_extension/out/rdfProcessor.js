"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.RDFProcessor = void 0;
const N3 = __importStar(require("n3"));
//import type { NamedNode } from '@rdfjs/types';
class RDFProcessor {
    //private prefixes: N3.Prefixes<NamedNode<string>> = {};
    constructor() {
        this.parser = new N3.Parser({ format: 'application/n-quads' });
        this.store = new N3.Store();
    }
    /**
     * Validate a Turtle/RDF file
     * @param fileContent File content to validate
     * @returns Validation result
     */
    async validate(fileContent, format) {
        const errors = [];
        // Set parser format if specified
        if (format) {
            this.parser = new N3.Parser({ format });
        }
        try {
            // Clear previous store
            this.store.removeQuads(this.store.getQuads(null, null, null, null));
            // Wrap parsing in a Promise
            await new Promise((resolve, reject) => {
                this.parser.parse(fileContent, (error, quad) => {
                    if (error) {
                        errors.push(error.message);
                        reject(error); // Reject the promise when an error occurs
                    }
                    else if (quad) {
                        this.store.addQuad(quad);
                        //} else if (prefixes) {
                        // Store prefixes from the parsed data
                        //this.prefixes = { ...this.prefixes, ...prefixes };
                    }
                    else {
                        // Resolve when parsing is done
                        resolve();
                    }
                });
            });
            return {
                valid: errors.length === 0,
                errors: errors.length > 0 ? errors : undefined
            };
        }
        catch (err) {
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
    async serialize(format = 'trig') {
        const writer = this.createWriter(format);
        // Add prefixes to the writer
        //writer.addPrefixes(this.prefixes);
        writer.addQuads(this.store.getQuads(null, null, null, null));
        return new Promise((resolve, reject) => {
            writer.end((error, result) => {
                if (error) {
                    reject(error);
                }
                else {
                    resolve(result);
                }
            });
        });
    }
    /**
     * Create a writer based on the specified format
     * @param format Serialization format
     * @returns N3 Writer
     */
    createWriter(format) {
        switch (format.toLowerCase()) {
            case 'n3':
                return new N3.Writer({ format: 'N3' });
            case 'nquads':
                return new N3.Writer({ format: 'N-Quads' });
            case 'jsonld':
                return new N3.Writer({ format: 'JSON-LD' });
            case 'turtle':
                return new N3.Writer({ format: 'Turtle' });
            case 'trig':
            default:
                return new N3.Writer({ format: 'TriG' });
        }
    }
}
exports.RDFProcessor = RDFProcessor;
//# sourceMappingURL=rdfProcessor.js.map