module.exports = {
    name: 'gbad-vsce',
    version: '0.0.2-prerelease.3',
    description: '',
    main: 'out/extension.js',
    engines: {
      vscode: '^1.98.0'
    },
    "categories": [
        "Other"
    ],
    "activationEvents": [
        "onLanguage:trig",
        "onLanguage:nq",
        "onLanguage:turtle",
        "onLanguage:nt"
    ],
    scripts: {
      "vscode:prepublish": "npm run compile",
      "compile": "tsc -p ./",
      "watch": "tsc -watch -p ./"
    },
    //activationEvents: [
    //    "onCommand:uuid-generator.generateUUID",
    //    "onCommand:rdfIndexLookup.checkGraph"
    //],
    contributes: {
      commands: [
        {
          command: "uuid-generator.generateUUID",
          title: "Generate UUIDv5 from Selection"
        },
        {
          command: "rdfValidator.validate",
          title: "GBAD: Validate and Serialize"
        },
        {
          command: "rdfIndexLookup.checkGraph",
          title: "GBAD: Triplestore Index Lookup"
        }
      ],
      menus: {
        "editor/context": [
          {
            command: "uuid-generator.generateUUID",
            group: "uuid@1",
            when: "editorHasSelection"
          },
          {
            command: "rdfValidator.validate",
            group: "rdf@1"
          }
        ]
      },
      configuration: {
        title: "RDF Validator",
        properties: {
          "rdfValidator.outputFormat": {
            "type": "string",
            "default": "trig",
            "enum": ["trig", "nquads", "n3", "turtle", "jsonld"],
            "description": "Output serialization format"
          },
          "rdfIndexLookup.indexFilePath": {
            "type": "string",
            "default": "path/to/index.json",
            "description": "Path to the RDF index file."
          },
          "rdfIndexLookup.indexUriPath": {
            "type": "string",
            "default": "http://example.org/index",
            "description": "URI path for the RDF index."
          },
          "rdfIndexLookup.sparqlEndpoint": {
            "type": "string",
            "default": "http://localhost:3030/sparql",
            "description": "SPARQL endpoint URL for fetching RDF graphs."
          }
        }
      }
    },
    dependencies: {
      "n3": "^1.17.2",
      "node-fetch": "^2.6.7"
    }
  };
