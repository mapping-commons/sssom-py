PYTHON=python
SSSOM_VERSION_TAG=0.12.0
SSSOM_PY="https://raw.githubusercontent.com/mapping-commons/sssom/$(SSSOM_VERSION_TAG)/src/sssom_schema/datamodel/sssom_schema.py"
SSSOM_YAML="https://raw.githubusercontent.com/mapping-commons/sssom/$(SSSOM_VERSION_TAG)/src/sssom_schema/schema/sssom_schema.yaml"
SSSOM_JSON_SCHEMA="https://raw.githubusercontent.com/mapping-commons/sssom/$(SSSOM_VERSION_TAG)/project/jsonschema/sssom_schema.schema.json"
SSSOM_JSONLD_CONTEXT="https://raw.githubusercontent.com/mapping-commons/sssom/$(SSSOM_VERSION_TAG)/project/jsonld/sssom_schema.context.jsonld"
OBO_EPM_JSON="https://raw.githubusercontent.com/biopragmatics/bioregistry/main/exports/contexts/obo.epm.json"

all: test

EXTS = _datamodel.py .schema.json .context.jsonld .external.context.jsonld .yaml

all_schema: $(patsubst %,schema/sssom%, $(EXTS))

.PHONY: .FORCE

schema/%_datamodel.py: .FORCE
	wget $(SSSOM_PY) -O $@
schema/cliquesummary.py: schema/cliquesummary.yaml
	gen-py-classes $< > $@
schema/%.schema.json: .FORCE
	wget $(SSSOM_JSON_SCHEMA) -O $@
src/sssom/obo.epm.json:
	wget $(OBO_EPM_JSON) -O $@
schema/%.context.jsonld: .FORCE
	wget $(SSSOM_JSONLD_CONTEXT) -O $@
schema/%.yaml sssom/%.yaml: .FORCE
	wget $(SSSOM_YAML) -O $@

test:
	pip install --upgrade pip
	pip install --upgrade tox
	tox
	sh tests/tests.sh

deploy-dm: src/sssom/obo.epm.json
	
install:
	pip install .[test,docs]

pypi: test
	echo "Uploading to pypi. Make sure you have twine installed.."
	python setup.py sdist
	twine upload dist/*

.PHONY: lint
lint:
	pip install tox
	tox -e lint

.PHONY: mypy
mypy:
	pip install tox
	tox -e mypy

# .PHONY: sphinx
# sphinx:
# 	cd sphinx &&\
# 	make clean html

# .PHONY: deploy-docs
# deploy-docs:
# 	cp -r sphinx/_build/html/* docs/
