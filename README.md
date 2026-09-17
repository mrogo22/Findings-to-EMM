# Findings-to-EMM
This repository is built for the paper _From the Structure of Comparative Findings to Their Search: Assisting Process Analysts with Exceptional Model Mining_
## Overview

Process mining offers an expanding range of techniques for analyzing event data, but this also creates a large space of possible analyses that analysts cannot explore exhaustively. Supporting the systematic discovery of relevant findings therefore goes beyond providing additional analysis techniques. It requires an explicit representation of what constitutes a finding and which analytical choices determine it. We address this problem for comparative process findings by reconstructing findings from BPI Challenge 2020 reports and analyzing how findings produced from the same event data and analytical questions correspond and vary. Through constant comparative analysis, we identify seven components that distinguish comparative findings: process object, case selection, grouping criterion, group definition, target property, measure, and comparison. 

We then use this structure to specify process aware bounded searches with Exceptional Model Mining (EMM), in which selected components define the analytical scope while candidate group definitions are searched systematically. We evaluate this operationalization on BPI Challenge 2020 and 2019 data using numerical throughput, binary overspending, and ordinal conformance deviation targets. The resulting searches recover subgroups corresponding to findings reported by human analysts while also returning high ranking candidate findings for which no corresponding reported finding was identified. Enlarging the admissible description space further changes which candidate findings can be discovered, illustrating how the derived structure makes the choices that bound comparative subgroup search explicit and executable.

Here we provide the reconstructed BPIC 2020 findings, their provenance, question mappings, and structural decomposition of comparative findings. The BPIC 2019 records used in the experiments are provided separately and include only conformance-related comparative findings, without structural decomposition. We provide the complete code for the experiments, together with an extended version of the pysubgroup (Lemmerich & Becker, 2018) package use to implement our proposed quality functions for EMM. 

## Prerequisites

You shoud unzip the datasets prior to running the code. The package requires no installation, provided that the file structure is maintained as given here. 
