# Source: https://github.com/ICA-EGAD/RiC-O/blob/5be6ec4e94d6735a7b5200cccdfca723b0c4d573/.github/workflows/consistency_check.py
# Imported on 2025-04-15, modified with Gemini 2.0 Flash and manually

"""
Simple consistency checker (and parser) for the RiC-O ontology, using the
Owlready2 module (pip install owlready2) and the Hermit reasoner built into
it. Requires that a JVM (Java virtual machine) be installed.

The environment variable ONTOLOGY_PATH can be either a path to an actual
version of the RiC-O ontology (expecting a .rdf file suffix and for the file
name to begin with 'RiC') or a directory. In the latter case the directory will
be searched for a file with a .rdf suffix whose name begins with 'RiC'; with
the current structure of the RiC-O ontology repository, this will allow for the
consistency checker to continue to work upon changes of version.
"""

from pathlib import Path
from sys import exit as sys_exit
import argparse

from rdflib import Graph
from owlready2 import get_ontology, sync_reasoner  # type: ignore
from owlready2.base import (  # type: ignore
    OwlReadyInconsistentOntologyError,
    OwlReadyOntologyParsingError)

class OntologyChecker():
    def __init__(self, ontology_path: str):
        given_path = self._check_path(ontology_path)

        # Load Turtle file
        g = Graph()
        g.parse(given_path)

        # Build the output filename
        serialized_path = given_path.with_name(f"{given_path.stem}_serialized.nt")
        
        # Save as N-Triples
        g.serialize(destination=serialized_path, format="nt", encoding="utf-8")

        self.ontology_path = serialized_path

    def _get_ontology_path(self):
        return self.ontology_path

    def _check_path(self, ontology_path: str) -> Path:
        given_path = Path(ontology_path)
        if not given_path.is_file():
            sys_exit(f"No file found at the provided path: {ontology_path}")
        return given_path

    def check_consistency(self) -> None:
        """
        Obtains the RiC-O ontology by means of the ONTOLOGY_PATH environment
        variable, parses it, and checks its consistency (using the Hermit reasoner
        built into Owlready2).
        """
        ontology_path = self._get_ontology_path()
        try:
            ontology = get_ontology(str(ontology_path)).load()
        except OwlReadyOntologyParsingError as e:
            sys_exit(f"Could not parse {ontology_path} to an ontology! Error: {e}")
        with ontology:
            try:
                sync_reasoner(debug=0)
            except OwlReadyInconsistentOntologyError:
                sys_exit("Ontology is inconsistent!")
        inconsistent_classes = list(ontology.inconsistent_classes())
        if inconsistent_classes:
            sys_exit("Ontology is not itself inconsistent, but has inconsistent "
                    f"classes: {inconsistent_classes}")
        print("Ontology is consistent!")

def get_ontology_path_from_cli():
    """
    Retrieves the ontology path from a command-line argument.

    Returns:
        str: The ontology path provided as a command-line argument.
            Exits the program if the argument is not provided.
    """
    parser = argparse.ArgumentParser(description="Script requiring an ontology path.")
    parser.add_argument("ontology_path", help="Path to the ontology file.")
    args = parser.parse_args()
    return args.ontology_path

if __name__ == "__main__":
    ontology_path = get_ontology_path_from_cli()
    ontology_checker = OntologyChecker(ontology_path)
    print(f"The ontology path provided is: {ontology_path}")

    ontology_checker.check_consistency()
    # You can now use the 'ontology_path' variable in your script
    # instead of relying on the environment variable.
    # For example:
    # with open(ontology_path, 'r') as f:
    #     # Process the ontology file
    #     pass
