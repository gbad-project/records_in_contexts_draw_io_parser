# Initialize npm project
npm init -y --init-module ./npm-init.js

# Install VS Code extension types and TypeScript
npm install --save-dev typescript @types/vscode @vscode/vsce @types/uuid @types/n3 @types/node-fetch

# Install uuid library
npm install uuid @types/uuid n3 node-fetch

# Compile the extension
npm run compile

# Package
vsce package
