module.exports = {
    name: 'gbad-vsce',
    version: '0.0.1',
    description: '',
    main: 'out/extension.js',
    engines: {
      vscode: '^1.96.0'
    },
    scripts: {
      "vscode:prepublish": "npm run compile",
      "compile": "tsc -p ./",
      "watch": "tsc -watch -p ./"
    },
    //activationEvents: [
    //    "onCommand:uuid-generator.generateUUID"
    //],
    contributes: {
      commands: [
        {
          command: "uuid-generator.generateUUID",
          title: "Generate UUIDv5 from Selection"
        }
      ],
      menus: {
        "editor/context": [
          {
            command: "uuid-generator.generateUUID",
            group: "uuid@1",
            when: "editorHasSelection"
          }
        ]
      }
    }
  };
