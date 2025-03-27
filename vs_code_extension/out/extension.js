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
exports.activate = activate;
exports.deactivate = deactivate;
// extension.ts
const vscode = __importStar(require("vscode"));
const uuid_1 = require("uuid");
const rdfProcessor_1 = require("./rdfProcessor");
function activate(context) {
    let disposable = vscode.commands.registerCommand('uuid-generator.generateUUID', () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active text editor');
            return;
        }
        const selection = editor.selection;
        if (selection.isEmpty) {
            vscode.window.showErrorMessage('Please select some text first');
            return;
        }
        const selectedText = editor.document.getText(selection);
        // Using URL namespace as default
        const generatedUuid = (0, uuid_1.v5)(selectedText, uuid_1.v5.URL);
        editor.edit(editBuilder => {
            editBuilder.replace(selection, generatedUuid);
        });
    });
    // New RDF Validation command
    const rdfProcessor = new rdfProcessor_1.RDFProcessor();
    let rdfValidateDisposable = vscode.commands.registerCommand('rdfValidator.validate', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active text editor');
            return;
        }
        const document = editor.document;
        const inputFormat = document.languageId === 'turtle' ? 'text/turtle' :
            document.languageId === 'n3' ? 'text/n3' :
                document.languageId === 'trig' ? 'application/trig' :
                    'application/n-quads';
        //vscode.window.showInformationMessage('Input Format: ' + inputFormat);
        const fileContent = document.getText();
        // Validate the file
        const validationResult = await rdfProcessor.validate(fileContent, inputFormat);
        if (validationResult.valid) {
            // Get configured output format
            const config = vscode.workspace.getConfiguration('rdfValidator');
            const outputFormat = config.get('outputFormat', 'turtle');
            try {
                // Serialize the graph
                const serializedOutput = await rdfProcessor.serialize(outputFormat);
                // Create a new preview column next to the current editor
                const column = editor.viewColumn === vscode.ViewColumn.One
                    ? vscode.ViewColumn.Two
                    : vscode.ViewColumn.One;
                //vscode.window.showInformationMessage(fileContent);
                //vscode.window.showInformationMessage(serializedOutput);
                // Create a new untitled document with serialized output
                vscode.workspace.openTextDocument({
                    content: serializedOutput,
                    language: outputFormat
                }).then(doc => {
                    vscode.window.showTextDocument(doc, {
                        viewColumn: column,
                        preview: true,
                        preserveFocus: true // Keep original file open
                    });
                });
                vscode.window.showInformationMessage('RDF Validated Successfully');
            }
            catch (error) {
                vscode.window.showErrorMessage('Serialization Error: ' + (error instanceof Error ? error.message : String(error)));
            }
        }
        else {
            // Show validation errors
            vscode.window.showErrorMessage('RDF Validation Failed: ' +
                (validationResult.errors?.join(', ') || 'Unknown error'));
        }
    });
    context.subscriptions.push(disposable);
    context.subscriptions.push(rdfValidateDisposable);
}
function deactivate() { }
//# sourceMappingURL=extension.js.map