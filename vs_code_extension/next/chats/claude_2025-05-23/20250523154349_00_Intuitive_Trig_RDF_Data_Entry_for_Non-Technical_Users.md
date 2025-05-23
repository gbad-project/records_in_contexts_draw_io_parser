# Intuitive Trig RDF Data Entry for Non-Technical Users

*Exported on: 5/23/2025, 3:43:49 PM*

## User

simple straightforward vs code extension that - editor of trig files. it  breaks new ground in the adoption of trig rdf data entry by non technical specialists by bringing a familiar, very intuitive form based interface but with the power and infinite flexibility of quad based data format under the hood. its role is to mask out some of the technical bits (eg the need to worry about prefixes, uris, syntax validity) while not restricting quad construction in any way. here is how it renders:
- the extension "lives" as a separate icon in vs code left sidebar. its behavior mimics Files tab in all respects but with a few differences. lets call this tab "Datasets" moving forward. namely, when showing files and folders it only shows any *.trig files (including in any nested subfolders). list of allowed file extensions can be edited  in extension setting.
- when clicking trig file, it validates as rdf trig (empty file is also fine). if fails, just shows a page that says that it failed to render and offers to open in stock text editor - the page can mimic exactly the page that opens when trying to open a malformed svg with stock vs code tooling.
- if trig is valid, actually show the same page but it says that validation was successful, please use left side pane for further editing. and offers to open with text editor as well.
- also upon valid trig, in the left pane the following is loaded. its contents are refreshed each time trig file is saved. the selected trig file becomes a collapsible under which you see the following category: Named graphs. it is a collapsible which lists named graphs. "default"  item is shown if there are any triples outside of any named graphs. how a list item behaves, i will describe later below.
- when clicking "named graphs" category, "add a new named graph" form opens. it uses a unified "add quad" form format which i will now describe. it is abstract in a sense that will then be implemented by more specific form types. it mimics a stock vs code settings page (e.g., grey keys, textboxes for values).  "add {specific form name} quad" is showing as if it was a named group of settings. the keys and values are specific to implementing form. after all keys and values there is a button "preview quad", when clicking it then all value text boxes become read only and preview button unclickable, and form-specific logic is triggered and based on this resulting quad is shown below nquads serialized and as view only. if resulting quad construction logic fails, a popup error is shown and nothing happens. below quad preview, there are three buttons: "return to editing", "edit nquad directly", "add to dataset". return to editing just clears previewed quad and unlocks the preview button. edit quad directly makes quad editable but value textboxes and preview button remain locked and return to editing is also locked now, whereas direct edit button now reads "undo direct edits". finally, when clicking "add" button, resulting quad is attempted to be added to dataset loaded from trig file. if ok, success popup message is shown and form clears, else just show error message and nothing else happens. sometimes form specific logic will construct several quads, then they must be all processed in single transaction!
- specifics of "add named graph" form: available keys include: authority name, date in yyyy-mm-dd, specific and fragment, to follow the abnf of tagURI = "tag:" authorityName "," date ":" specific [ "#" fragment ] . quad construction logic: based on filled values a valid tag uri is attempted to be constructed, which is then appended to a named graph base uri which is set in extension settings (default to "https://data.test.gbad.ca/graphs/"). then an empty quad is constructed with this as context term. 
- Named graphs collapsible - list item: represents a single named graph. when clicked, an "edit named graph form". it implements a unified "edit quad" form which i will now describe. visually it mimics the corresponding add form. the form uniquely locates the quad in dataset according to add form logic (or it can be multiple quads if that's what quad construction logic says) and parses it (them) to fill fields of corresponding add form. if parsing is successful, then everything is loaded and form behaves like add form just that "add" button is replaced with "replace quad" button which - in single transaction! - removes the old quad and adds new one. if parsing was not successful, the form still loads but key value boxes, preview button, return to editing, and direct edit button are all greyed out (and the button names read - "form editing unavailable") and this enters direct editing automatically, so user can only directly edit quad and press "replace quad" button. and on the opposite side of replace quad there is also now (in contrast with add form) red color text button "remove quad" that is always clickable but request confirmation upon clicking.
- it is of note that both add quad and edit quad forms upon finalizing transaction, write this to trig file - and it's of utmost importance to make it such that vs code recognized this simply as text file edits, so if we open trig file in text editor concurrently, we can undo redo changes easily. by the way, if upon any action the extension sees that trig file was modified externally, it gives user a default vs code pop up that prompts whether they want to overwrite or not. whenever multiple quads must be processed, then they must be all processed in a single transaction!
- going back to named graph list item. it is also itself a collapsible that when expanded contains a category - "Subjects". this obviously lists subjects. when this category is clicked, an "add subject form" is opened which implements unified add form. specifics: available key values include, subject prefix,  subject local name, class prefix, class local name, rdfs label. prefixes can be added globally in extension settings. when clicked preview, it validates if prefixes are recognized from extension settings and pops up error if not and instructions for user how to set up one in extension settings. quad construction: subject prefix+local name ; rdf:type ; class prefix+local name ; named graph uri as context. and another quad with rdfs label.
-  now finally, each item on list of subjects is also a collapsible and contains a category: "Predicate object pairs". it obviously lists predicate object pairs. when clicking the category, an add form is opened (obviously implements unified add quad interface). specifics: key values available in add form include, predicate prefix, predicate local name, object type (selector: prefix-local name, full uri, literal), then accordingly object prefix, object local name, or object full uri, or object literal. quad construction is done accordingly.
- for subjects and predicate-object pairs editing functionality is of course also implemented accordingly using unified edit quad approach. if upon attempting to edit any complex entities are found (eg, empty nodes with nested stuff), just default to direct editing.
now i ask you to please respond with a single paragraph -  are you good or not to proceed to code writing? any parts seem obscure, badly specified, or otherwise require clarification/review?

## Claude

I understand the overall architecture and can proceed with coding this VS Code extension. The specification is quite comprehensive - you want a form-based RDF/Trig editor that presents a familiar interface while maintaining the flexibility of quad-based data underneath. The key components are clear: a custom sidebar mimicking the Files explorer but filtered for .trig files, validation workflows, hierarchical collapsible views (Named Graphs → Subjects → Predicate-Object Pairs), and unified add/edit forms with preview functionality that writes changes as standard text edits to maintain VS Code's undo/redo compatibility. A few minor clarifications would be helpful during implementation: the exact error handling strategy when quad transactions partially fail, the specific format for the extension settings schema (especially for prefixes and base URIs), and whether there are any constraints on the RDF libraries you'd prefer for parsing/serialization, but these details can be worked out as we build. The core concept of masking technical RDF complexity while preserving full expressiveness through the form interface is well-defined and achievable.



## User

A few minor clarifications:the exact error handling strategy when quad transactions partially fail: ideally partial failing should not arise because all quad add/replace/remove operations must be done in single transactions (either success of fail completely - just show a popup and do nothing meaning that all form contents should remain as is). should partial failing happen anyway unexpectedly, at any point, adhere to vs code stock file handling operations as closely as possible because remember that "transactions" in the context of this extension ultimately result (in case of success) simply in trig file write operations, which as specified should be implemented in such a way that vs code recognizes them as native text file edits and can flawlessly undo/redo - this is really important and must guide you when implementing error handling in within-extension transactions.the specific format for the extension settings schema: overall choose the most straighforward/convenient. specifically for prefixes, make a textarea where user can input any trig that will be automatically added to all datasets opened with extension. for base named graph uri, just make a textbox where user can input a uri and validate if it's valid uri on input.whether there are any constraints on the RDF libraries you'd prefer for parsing/serialization: no constraints

## Claude

Perfect, those clarifications resolve the key implementation details. The atomic transaction approach with complete success/failure aligns well with VS Code's native file handling, and treating all operations as standard text edits will ensure seamless undo/redo integration. Using a textarea for prefix definitions that get automatically prepended to datasets is elegant and straightforward, and URI validation for the base graph setting is simple to implement. With no constraints on RDF libraries, I can choose appropriate ones for robust Trig parsing and N-Quads serialization. I'm ready to proceed with coding this extension - the specification is now complete and clear enough to build a functional VS Code extension that delivers the intuitive form-based RDF editing experience you've described.


**Code Artifact:** `01_VS Code Extension Package Configuration` (Code)
*See separate file with corresponding timestamp prefix*


**Code Artifact:** `02_Main Extension Entry Point` (Code)
*See separate file with corresponding timestamp prefix*


**Code Artifact:** `03_Trig Datasets Tree Provider` (Code)
*See separate file with corresponding timestamp prefix*



## User

continue where you left

## Claude


**Code Artifact:** `01_Trig Datasets Tree Provider` (Code ∙ Version 2)
*See separate file with corresponding timestamp prefix*


**Code Artifact:** `02_Trig Document Content Provider` (Code)
*See separate file with corresponding timestamp prefix*


**Code Artifact:** `03_Form Panel Provider for Editing` (Code)
*See separate file with corresponding timestamp prefix*



