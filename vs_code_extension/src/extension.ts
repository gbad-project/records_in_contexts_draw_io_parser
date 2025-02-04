// extension.ts
import * as vscode from 'vscode';
import { v5 as uuidv5 } from 'uuid';

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

    context.subscriptions.push(disposable);
}

export function deactivate() {}
