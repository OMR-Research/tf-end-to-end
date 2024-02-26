#!/bin/bash

# Adapted from "semantic_conversor.sh", originally authored by Calvo-Zaragoza

java -cp converter/omr-3.0-SNAPSHOT.jar es.ua.dlsi.im3.omr.encoding.semantic.SemanticImporter $*
