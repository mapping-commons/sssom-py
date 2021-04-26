# SSSOM Command Line Client: Design

This document is for internal use only. The goal is to sketch the design of the SSSOM CLI in terms of main methods.

- Basic design decisions
- Core methods (convert, validate, merge)
- Support method (deduplicate)

### Basic design decisions

- IO: Flexible vs fixed formats
  1. We build `sssom` entirely around the table format and consider all other formats one-way exports. This means, concretely, that for all commands other than `convert`, `-i` and `-o` parameters are fixed to be sssom tsv - the advantage here is simplicity for handling of errors, but also that we don't need to overload the main CLI parameters with `--output-format` or `--input-format` parameters.
  2. We keep `-i` and `-o` variables flexible such as ROBOT and allow a (potentially faulty) inference step from file extension to format. The disadvantage is that we need to extend all main methods to deal with formats. You could allow external mode (table and metadata separately) only in convert function, and require embedded mode everywhere else. This also means we don't need the context parameter anywhere but in convert.
- IO: Allow both embedded and non-embedded mode. The complexity here is to handle conflicts: what if an external metadata block is supplied, but the SSSOM file already has its own? This needs to be documented carefully.
- Context - is it purely for the sake of the curie map?
- IO: Make `-o` entirely optional and print result to stdout? This could allow | style chaining of commands!
- IO: remove `-i` in favour of nothing? `sssom convert a | sssom deduplicate |`
- IO: more disciplined seperation of data model and serialisation in convert

### General CLI stuff
The following are inspired from https://clig.dev/.

- Help
  - Display help text when passed no options, the -h flag, or the --help flag.
  - Add github issue tracker location for feedback (Make it effortless to submit bug reports. One nice thing you can do is provide a URL and have it pre-populate as much information as possible.)
  - Lead with examples
- Display output on success, but keep it brief (add `--quiet` option)
- Catch errors and rewrite them for humans
- Have full-length versions of all flags, but one length flags only for the most important ones
- standard flags: `-d/--debug`, `-f/--force`, `-h/--help`, `-v/--verbose`, `--version`, `-q/--quiet`
- If input or output is a file, support - to read from stdin or write to stdout.



### Core methods

- convert
- merge
- validate

#### merge

Purpose: Take as an input 1 or more SSSOM files and produce a merged output.

Design considerations:
- If conflicting metadata or curie_maps: First-come principle. For all subsequent conflicts a warning is printed.
- Row identity determined by all the metadata available (so potential for duplicates)

```
sssom merge -i t1.sssom.tsv -i t2.sssom.tsv ... -o t_merged.sssom.tsv
```

#### convert

Purpose: Take as an input exactly 1 SSSOM file and convert it to the specified format.

```
sssom convert -i t.sssom.tsv --from-format tsv --to-format owl -o t_converted.sssom.*
```
