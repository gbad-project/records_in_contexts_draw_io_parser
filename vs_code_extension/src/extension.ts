// extension.ts
import * as vscode from 'vscode';
import { v5 as uuidv5 } from 'uuid';
import { RDFProcessor } from './rdfProcessor';

export function activate(context: vscode.ExtensionContext) {
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
        const generatedUuid = uuidv5(selectedText, uuidv5.URL);

        editor.edit(editBuilder => {
            editBuilder.replace(selection, generatedUuid);
        });
    });

    // New RDF Validation command
    const rdfProcessor = new RDFProcessor();
    let rdfValidateDisposable = vscode.commands.registerCommand('rdfValidator.validate', async (): Promise<void> => {
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
        let fileContent = document.getText();

        // Assume user wants to only check selection if text selected
        const selection = editor.selection;
        if (!selection.isEmpty) {
            fileContent = editor.document.getText(selection);
        }

        // Validate the file
        const validationResult = await rdfProcessor.validate(fileContent, inputFormat);

        if (validationResult.valid) {
            // Get configured output format
            const config = vscode.workspace.getConfiguration('rdfValidator');
            const outputFormat = config.get<string>('outputFormat', 'turtle');
            
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
            } catch (error) {
                vscode.window.showErrorMessage('Serialization Error: ' + (error instanceof Error ? error.message : String(error)) );
            }
        } else {
            // Show validation errors
            vscode.window.showErrorMessage(
                'RDF Validation Failed: ' + 
                (validationResult.errors?.join(', ') || 'Unknown error')
            );
        }
    });

    context.subscriptions.push(disposable);
    context.subscriptions.push(rdfValidateDisposable);
}

export function deactivate() {}
