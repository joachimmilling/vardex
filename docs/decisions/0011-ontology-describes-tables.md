# 0011 Declare the pack's ontology over existing tables; never create them

Status: Accepted
Date: 2026-10-12

## Context

A model writing SQL must know what a row means and how tables join. Column descriptions say
what a column holds, not that `fct_accounts` holds one set of accounts per company, year and
kind, or that it joins to `dim_companies` on `orgnr` and to nothing else. Later features, such as
actions on objects and permissions that follow the data, need the same knowledge. Ontology
products often own the data as well: they load it into their own store.

## Options

- No ontology: let the model infer objects and joins from names and descriptions.
- An ontology that owns its storage: objects are defined first, and Vardex creates and fills
  their tables.
- An ontology that describes existing tables: each object names a table, a key column and a
  label; each link names the column that holds another object's key.

## Decision

Each pack may declare `warehouse/ontology.yaml` with objects and many-to-one links over tables
that already exist. Vardex never creates or changes a table for it. On load, every object and
link must name described models and columns; on every ingest, and with `vardex check` against
any warehouse, every key must be unique and present, and every link must find its object.

## Consequences

- The model is told how to join, in the pack's words, and the check proves it on real data.
- A company's ontology sits on top of the warehouse its data team runs, unchanged.
- Many-to-many links, such as one through a bridge table, need a later extension.
- An object is only as good as its table; if the table is wrong, the ontology says so loudly
  rather than fixing it.

## When to revisit

When actions are added to objects, or when packs need links through bridge tables.
