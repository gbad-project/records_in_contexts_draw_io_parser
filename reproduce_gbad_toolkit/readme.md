# README

Author: Pavel Zhelnov

## Executive summary

On June 2nd, 2025 (UTC-4), I tried to reproduce the steps outlined in the original [GBAD Implementation Toolkit](https://github.com/gbad-project/.github/raw/5f3d62a6e16eba460dacfddeadd1a8bbf1a07498/GBAD_Toolkit_Fox_Huang_Luchin_2025-04-10.pdf) (version 2025-04-10) and failed to do it – despite being one of the authors of the scripts involved and a consultant on the Toolkit itself. This README file documents my attempt.

**Why is it irreproducible and what can be done to fix this?** The most important reason is that I did not attempt to reproduce the steps necessary when consulting on the Toolkit and did not provide adequate guidance.

At this time, both the `map_schema.py` and `map_rml.py` files need to be modifed manually to support custom data. Note that Python coding skills may be needed. Please [open an issue](https://github.com/gbad-project/records_in_contexts_draw_io_parser/issues/new) if you are reading this and need help or reach out using the contacts indicated on <https://github.com/gbad-project>

## Directory contents

[This directory](https://github.com/gbad-project/records_in_contexts_draw_io_parser/tree/gbad-project/reproduce_gbad_toolkit) intends to reproduce exactly the conversion workflow as described in the GBAD Implementation Toolkit: https://github.com/gbad-project/.github/raw/5f3d62a6e16eba460dacfddeadd1a8bbf1a07498/GBAD_Toolkit_Fox_Huang_Luchin_2025-04-10.pdf

`source` subdirectory contains the relevant files downloaded as is by clicking the links in the Toolkit PDF. Because the links referenced in the Toolkit contain the specific commit hashes, they provide the exact same versions over time.

Because I had failed to specify the device / environment / dependencies requirements in the Toolkit, I had to improvize at reproduction.

`source_edited` subdirectory contains the files from source that I had to edit to make the reproduction work.

`results` contains the outputs obtained after running the scripts and data from the source subdirectory.

## Reproduction notes

Reproduction set-up: Ubuntu 24.10 (aarch64) via limactl version 1.0.7 under macOS Sequoia (Apple M4).

Assuming the following paths:

```bash
git clone https://github.com/gbad-project/records_in_contexts_draw_io_parser.git ~/records_in_contexts_draw_io_parser
cd ~/records_in_contexts_draw_io_parser/reproduce_gbad_toolkit
```

I created a new [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) environment with the latest version of Python using the following Bash commands:

```bash
conda create -n reproduce_gbad_toolkit python --yes
conda activate reproduce_gbad_toolkit
```

I then added Poetry to this environment to pin dependencies (to ensure the reproducibility of this reproduction):

```bash
pip install poetry
poetry init --no-interaction
```

I then tried to run the scripts immediately, and whenever there was an error due to a missing package, I would just run `poetry add <package>`.

## Run Parser Scripts

The Toolkit specifies that three scripts must be run:

> **Draw_io_parser.py** \
> Parses draw.io file and generates .rml, an intermediary .ttl file representing visual schema.
>
> **Map_schema.py** \
> Applies preprocessing to .csv if necessary (e.g., separate/merge columns or replace/clean some values according to pre-specified rules). Links the preprocessed dataset .csv to .rml.
>
> **Map_rml.py** \
> Maps the .rml file to create the final .ttl file. Applies post- processing to .ttl if necessary (e.g., remove or add some triples based on a set of predefined SPARQL queries).

Here is the approximate sequence in which I tried to run scripts (trying to follow the Toolkit, but I am of course biased because I already know the sequence).

### ✅ 1\. Draw_io_parser.py

> Parses draw.io file and generates .rml, an intermediary .ttl file representing visual schema.

```bash
COMMAND="python source/draw_io_parser.py"
script -c "$COMMAND" logs/draw_io_parser.log
```

This froze immediately because the script apparently anticipated some arguments. I reran the `COMMAND` with the `--help` argument and was able to get some instructions:

```bash
COMMAND="python source/draw_io_parser.py --help"
script -c "$COMMAND" logs/draw_io_parser.log
```

Analyzing these did not help me understand the usage. I had therefore to turn to the original parser’s repository: https://github.com/williamsonrichard/records_in_contexts_draw_io_parser

In the README there, I read that the command should be run as `cat example.drawio | python draw_io_parser.py`, which I attempted (using the *.drawio file that I downloaded from the **Template** button in the **Create Visual Schema** section of the Toolkit):

```bash
DRAWIO_FILE="source/General ADD (Descriptions and Listings) to RiC-O Model_2025-03-10_PZ.drawio"
COMMAND="cat '$DRAWIO_FILE' | python source/draw_io_parser.py"
script -c "$COMMAND" logs/draw_io_parser.log
```

After some experimentation, the above command was successful at getting a couple of Deprecation Warnings and the following message:

> The following contains a space, but how to handle spaces in individual nodes has not been specified (spaces cannot be used in OWL IRIs): 'rr:template "KB/RecordSet/{REFD_FILE}"'. Use the -m/--metacharacter-substitute and -c/--capitalisation-scheme options to define how to handle spaces

I had to review the `--help` instructions again and could not help but notice the `-m=url` option (which I had once added myself). I tried the following:

```bash
COMMAND="cat '$DRAWIO_FILE' | python source/draw_io_parser.py -m=url"
```

After this, I received two deprecation warnings again and the following message:

> An arrow has label 'rdfs:label', which is not a known object property or datatype property

I had to review the *.drawio file using https://draw.io to understand what was going on. Before opening it, I copied it to `source_edited` subdirectory.

Apparently the `draw_io_parser.py` script did not support rdfs:label (which was due to that an earlier version of the script appears to have been referenced in the Toolkit), so I had to simply remove all arrows that contained it (together with the corresponding literal node). After this, I tried to run the script again:

```bash
DRAWIO_FILE="source_edited/General ADD (Descriptions and Listings) to RiC-O Model_2025-03-10_PZ.drawio"
COMMAND="cat '$DRAWIO_FILE' | python source/draw_io_parser.py -m=url"
script -c "$COMMAND" logs/draw_io_parser.log
```

This produced the output but printed it in the shell, so I had to go back to the original Draw.io parser repository to review the README there again. After this, I edited the command again:

```bash
COMMAND="cat '$DRAWIO_FILE' | python source/draw_io_parser.py -m=url > results/example.owl"
```

An `example.owl` file was generated. Neither an .rml nor a .ttl file was generated, which suggests that this statement in the Toolkit is incorrect.

I have then tried to convert the other *.drawio file, which I obtained from the **Draw.io File** button in the **Create Visual Schema** section of the Toolkit.

```bash
DRAWIO_FILE="source/CA1853 Chest Disease Service.drawio"
COMMAND="cat '$DRAWIO_FILE' | python source/draw_io_parser.py -m=url > results/CA1853.owl"
script -c "$COMMAND" logs/draw_io_parser.log
```

`CA1853.owl` was successfully generated.

### ❌ 2\. Map_schema.py

> Applies preprocessing to .csv if necessary (e.g., separate/merge columns or replace/clean some values according to pre-specified rules). Links the preprocessed dataset .csv to .rml.

It was not clear where I can get a sample *.csv file. So just tried to run the script:

```bash
COMMAND="python source/map_schema.py"
script -c "$COMMAND" logs/map_schema.log
```

I got a `ModuleNotFoundError: No module named 'rdflib'`, so I ran `poetry add rdflib`.

This gave me an error:

```
For rdflib, a possible solution would be to set the `python` property to ">=3.13,<4.0.0"
```

So I edited this in `pyproject.toml`. This does not relate to the conversion logic but still is an unwarranted side effect (related to the use of the newest Python 3.13 in this case – the conversion pipeline was being developed using Python 3.9–3.12). After this edit Poetry successfully added the rdflib dependency, so I tried to run `$COMMAND` again.

I got a `ModuleNotFoundError: No module named 'pandas'`, so I ran `poetry add pandas` (successfully this time) and reran the command again.

I received two Syntax Warnings and an Error:

```
~/records_in_contexts_draw_io_parser/reproduce_gbad_toolkit/source/map_schema.py:612: SyntaxWarning: invalid escape sequence '\s'
  uriref_str = re.sub('\s+', ' ', uriref_str)
~/records_in_contexts_draw_io_parser/reproduce_gbad_toolkit/source/map_schema.py:615: SyntaxWarning: invalid escape sequence '\s'
  sin_predicate = re.sub(f"^{str(predicate)}\s+", "", uriref_str)
usage: map_schema.py [-h] schema [source]
map_schema.py: error: the following arguments are required: schema
```

I was thus obvious that a `schema` argument had to be specified, with an optional `[source]` argument. How these work, I had no idea from the Toolkit.

No other reference where I could look this up was listed, so the reproduction was halted at this point without exploring the source code, which was not the intention of the Toolkit.

### ❌ 3\. Map_rml.py

> Maps the .rml file to create the final .ttl file. Applies post- processing to .ttl if necessary (e.g., remove or add some triples based on a set of predefined SPARQL queries).

I still wanted to do the 3rd step using the *.rml and *.ttl files already available from the repository but could not find links to them in the Toolkit.

Thus, this step could not be reproduced.
