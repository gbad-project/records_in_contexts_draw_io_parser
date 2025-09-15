# No frills RML Mapping

As a heads-up, this guide is not sufficiently "no frills" yet, but this is the best I could come up with so far. My apologies!

**All operating systems**

* Download an rmlmapper Java executable file (*.jar) from [GitHub](https://github.com/RMLio/rmlmapper-java/releases/).
  * You may opt to use the version `rmlmapper-7.0.0-r374-all.jar` as it has been previously tested on the Graph-Based Archival Description project – [direct link](https://github.com/RMLio/rmlmapper-java/releases/download/v7.0.0/rmlmapper-7.0.0-r374-all.jar).
* Install [Java 21 Development Kit (JDK)](https://www.oracle.com/java/technologies/downloads/) on your system.
  * Alternatively, set up [Conda](https://www.anaconda.com/docs/getting-started/miniconda/install#quickstart-install-instructions) is using this [environment.yml](../next/environment.yml).
* Run a command file from [this directory](./), depending on your desktop operating system: Windows, macOS, or Linux (e.g., Ubuntu).
* **Note:** Relative paths in the RML file are resolved against the location of the RML file.

**Windows**

_Guide not developed yet._

**macOS**

* Open Terminal and `chmod +x ./macos/rmlmapper.command`
* In Finder, double-click the command file.
  * A Terminal window will pop up.
* Drag and drop the RML mapper *.jar file you downloaded earlier onto the terminal window and press <kbd>Enter</kbd>.
  * It will be appended to `./.env` file thereafter, unless already defined in the file.
* Enter the base URI you want to use with your RML file.
  * It will be appended to `./.env` file thereafter, unless already defined in the file.
* Drag and drop `./test.rml` onto the terminal window and press <kbd>Enter</kbd>.
* If the conversion is successful, a `test.ttl` file should now appear alongside the RML file.
  * This file should contain the converted triples.

**Linus**

_Guide not developed yet._
