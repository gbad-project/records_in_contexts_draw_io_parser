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
    let rdfValidateDisposable = vscode.commands.registerCommand('rdfValidator.validate', () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active text editor');
            return;
        }

        const document = editor.document;
        const fileContent = document.getText();

        // Validate the file
        const validationResult = rdfProcessor.validate(fileContent);

        if (validationResult.valid) {
            // Get configured output format
            const config = vscode.workspace.getConfiguration('rdfValidator');
            const outputFormat = config.get<string>('outputFormat', 'turtle');

            // Serialize the graph
            const serializedOutput = rdfProcessor.serialize(outputFormat);

            // Create a new untitled document with serialized output
            vscode.workspace.openTextDocument({
                content: serializedOutput,
                language: outputFormat
            }).then(doc => {
                vscode.window.showTextDocument(doc);
            });

            vscode.window.showInformationMessage('RDF Validated Successfully');
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
