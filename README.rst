===================================================
PLifePred2: Peptide Half-Life Prediction and Design
===================================================

PLifePred2 is a standalone machine-learning toolkit for predicting peptide
half-life from amino-acid sequences, molecular SMILES, or PDB structures. It
also provides a design workflow for systematic single-residue substitution
scanning and enumeration of peptide-modification flag combinations.

The distribution is designed to run locally. Model files, feature lists, and
the vendored Mordred-community implementation are loaded from the project
directory, so the program does not download models or descriptors at runtime.

.. image:: https://img.shields.io/badge/Python-3.10%2B-blue.svg
   :target: https://www.python.org/
   :alt: Python 3.10 or newer

.. image:: https://img.shields.io/badge/License-GPL--3.0-green.svg
   :target: https://www.gnu.org/licenses/gpl-3.0.html
   :alt: GNU General Public License version 3


Overview
========

PLifePred2 offers four input modes and one integrated design option:

* **Natural sequence** -- predicts half-life from sequence-derived features,
  with all modification flags set to zero.
* **Modified sequence** -- predicts half-life from sequence-derived features
  and five user-selectable binary modification flags.
* **SMILES** -- calculates 2D molecular descriptors with the bundled
  Mordred-community source and predicts from a SMILES representation.
* **PDB structure** -- converts the supplied PDB molecule to SMILES, calculates
  Mordred descriptors, and predicts using the structure model.
* **Design mode** -- performs exhaustive single-residue substitution scanning
  or evaluates all 32 combinations of the five modification flags.

Prediction is the default operation. Add ``--design`` to run one of the two
design workflows. The same command-line script handles both operations.


Key Features
============

* Four prediction routes: natural sequence, modified sequence, SMILES, and PDB.
* FASTA files containing multiple peptide records are supported.
* Multiple input files can be supplied in a single run.
* SMILES files can contain multiple molecules, with optional molecule IDs.
* Multiple PDB files can be processed in one run.
* Natural design includes the wild-type peptide and every possible single
  amino-acid substitution at each residue position.
* Modified design evaluates all 32 binary combinations of the five flags.
* Design results are ranked by predicted half-life for each input peptide.
* Feature lists and model files are resolved relative to the script location.
* Intermediate descriptor files can be retained for inspection.

There is no application-level limit on the number of sequence records, SMILES
records, or PDB input files. Very large design jobs can nevertheless require
substantial memory, processing time, and disk space.


Project Contents and Required Files
===================================

Keep the model and support files in the locations expected by the script. The
following is the default layout referenced by ``plifepred2.py``::

   PLifePred2/
   |-- plifepred2.py
   |-- default_regression_models.py
   |-- comp/
   |   |-- pfeature_comp.py
   |   `-- ... Pfeature support files and data ...
   `-- models/
       |-- Sequence_model.sav
       |-- sequence_features.tsv
       |-- RFR_combined_model.pkl
       |-- smile_features.csv
       |-- Struture_model.pkl
       `-- mordred-community/
           `-- mordred/
               |-- __init__.py
               `-- ... vendored Mordred-community modules ...

Important files
---------------

* ``Sequence_model.sav`` -- pretrained sequence model used by natural and
  modified sequence prediction and both sequence design workflows.
* ``sequence_features.tsv`` -- ordered list of the 100 sequence descriptors
  selected for the sequence model.
* ``RFR_combined_model.pkl`` -- pretrained model used for SMILES prediction.
* ``smile_features.csv`` -- selected Mordred feature list for the SMILES model;
  it must contain a column named ``Feature``.
* ``Struture_model.pkl`` -- pretrained model pipeline used for PDB structure
  prediction. The filename is intentionally spelled ``Struture_model.pkl``;
  keep this name unchanged unless the script constant is also updated.
* ``models/mordred-community/`` -- vendored Mordred-community source used to
  calculate molecular descriptors. Keep this directory intact and unchanged.
* ``default_regression_models.py`` -- compatibility module required when the
  structure model is loaded. Keep it beside ``plifepred2.py``.
* ``comp/pfeature_comp.py`` -- Pfeature entry point used by the current script.
  The complete Pfeature installation and all of its required data/resources
  must also be present in the layout expected by Pfeature; copying this single
  Python file alone is not sufficient.

The paths above are relative to the directory containing
``plifepred2.py``, not the directory from which the command is run.
The script's default Pfeature path is ``comp/pfeature_comp.py``. If Pfeature is
stored elsewhere in your distribution, update ``DEFAULT_PFEATURE`` near the
top of the script to match that location.


Installation
============

Requirements
------------

* Python 3.10 or newer.
* NumPy.
* pandas.
* scikit-learn.
* joblib.
* RDKit for the ``smiles`` and ``structure`` modes.
* A complete local Pfeature installation for the ``natural`` and ``modified``
  sequence modes and for sequence design.

The script invokes Pfeature with the active Python interpreter. Therefore,
Pfeature's Python dependencies must be available in the same environment.
The SMILES and structure workflows require RDKit; sequence-only workflows do
not import RDKit.

Method 1 -- PyPI installation
-----------------------------

Install PLifePred2 directly from PyPI:

.. code-block:: bash

   pip install plifepred2

Method 2 -- GitHub installation with Conda
------------------------------------------

Clone the PLifePred2 repository. The ``environment.yml`` file is included in
the repository and creates the tested Conda environment with the required
Python packages:

.. code-block:: bash

   git clone https://github.com/raghavagps/plifepred2.git
   cd plifepred2
   conda env create -f environment.yml
   conda activate plifepred2

Ensure that the required model files, support directories, and complete
Pfeature installation are present in the project layout shown above before
running sequence-based workflows.

If RDKit is not needed, it may be omitted from the environment for runs that
use only ``natural`` or ``modified`` sequence modes.


Command-Line Usage
==================

General syntax
--------------

.. code-block:: bash

   python plifepred2.py --mode MODE --input INPUT [INPUT ...] --output OUTPUT [OPTIONS]

The script can be launched from another working directory. Model and feature
paths are resolved relative to the script, while relative input and output
paths are resolved from the current working directory.

Arguments
---------

+--------------------------------+----------------------------------------------+
| Argument                       | Description                                  |
+================================+==============================================+
| ``--mode``                     | Required. One of ``natural``, ``modified``,  |
|                                | ``smiles``, or ``structure``.                |
+--------------------------------+----------------------------------------------+
| ``-i``, ``--input``            | Required. One or more input files.           |
+--------------------------------+----------------------------------------------+
| ``-o``, ``--output``           | Output TSV file. Default:                    |
|                                | ``plifepred2_predictions.tsv``.              |
+--------------------------------+----------------------------------------------+
| ``--design``                   | Run design instead of ordinary prediction.   |
|                                | Available for ``natural`` and ``modified``.  |
+--------------------------------+----------------------------------------------+
| ``--modifications``,           | Comma-separated modification flags for       |
| ``--mods``                     | ``modified`` prediction mode.                |
+--------------------------------+----------------------------------------------+
| ``--workdir``                  | Directory for intermediate descriptor        |
|                                | files. Temporary storage is used by default. |
+--------------------------------+----------------------------------------------+
| ``-h``, ``--help``             | Display command-line help.                   |
+--------------------------------+----------------------------------------------+


Prediction Modes
================

1. Natural peptide sequence
---------------------------

Use ``--mode natural`` for unmodified peptide sequences. All five
modification flags are set to zero.

.. code-block:: bash

   python plifepred2.py  --mode natural --input peptides.fasta --output natural_predictions.tsv

2. Modified peptide sequence
----------------------------

Use ``--mode modified`` to supply one or more binary modification flags. The
following names are accepted:

+----------------+-----------------------------------------------+
| Flag           | Description                                   |
+================+===============================================+
| ``d``          | D-amino-acid modification flag                |
+----------------+-----------------------------------------------+
| ``ct``         | C-terminal modification flag                  |
+----------------+-----------------------------------------------+
| ``nt``         | N-terminal modification flag                  |
+----------------+-----------------------------------------------+
| ``cyc``        | Cyclization flag                              |
+----------------+-----------------------------------------------+
| ``ptm``        | Post-translational modification flag          |
+----------------+-----------------------------------------------+

Example using flag names:

.. code-block:: bash

   python plifepred2.py --mode modified --input peptides.fasta --modifications d,ct --output modified_predictions.tsv

The script also accepts the legacy numeric labels ``0`` to ``4``:

+----------------+-----------------+
| Numeric value  | Flag            |
+================+=================+
| ``0``          | ``d``           |
+----------------+-----------------+
| ``1``          | ``ct``          |
+----------------+-----------------+
| ``2``          | ``nt``          |
+----------------+-----------------+
| ``3``          | ``cyc``         |
+----------------+-----------------+
| ``4``          | ``ptm``         |
+----------------+-----------------+

For example, ``--mods 0,3`` is equivalent to ``--mods d,cyc``. If no flags
are supplied in modified prediction mode, all five flags remain zero.

3. SMILES input
---------------

Use ``--mode smiles`` to predict from molecular SMILES. Provide one molecule
per non-empty line. Each line can contain a SMILES string and, optionally, a
molecule identifier separated by whitespace::

   CCCCC(=O)NCC(=O)O Molecule_A
   CC(C)CC(N)C(=O)O Molecule_B

Example:

.. code-block:: bash

   python plifepred2.py --mode smiles --input molecules.smi --output smiles_predictions.tsv

Multiple SMILES files can be processed together:

.. code-block:: bash

   python plifepred2.py --mode smiles --input batch_1.smi batch_2.smi batch_3.smi --output smiles_predictions.tsv

The script canonicalizes valid input SMILES before writing them to the output.
Invalid SMILES cause the run to stop with an error identifying the input
record.

4. PDB structure input
----------------------

Use ``--mode structure`` to predict from one or more PDB files:

.. code-block:: bash

   python plifepred2.py --mode structure --input peptide_1.pdb peptide_2.pdb --output structure_predictions.tsv

**PDB input requirements:** provide a PDB containing the peptide or molecule
to be evaluated, not a complete protein-receptor complex. The current
conversion workflow uses RDKit, retains the largest connected molecular
fragment, removes explicit hydrogens, and converts that fragment to SMILES.
For a protein-ligand or protein-peptide complex, the largest fragment may be
the receptor rather than the intended peptide/ligand; such a complex should
not be supplied directly to this mode.


Design Mode
===========

Design mode is enabled with ``--design`` and is available only with
``--mode natural`` or ``--mode modified``.

1. Natural single-substitution scanning
---------------------------------------

For each input peptide, PLifePred2 generates:

* The wild-type sequence.
* Every possible single-residue substitution at every position, replacing
  the original residue with each of the other 19 standard amino acids.

Only one position is changed in each mutant. It does not generate
multi-position combinatorial mutants.

The total number of variants for a peptide of length ``L`` is::

   Number of variants = 1 + (19 x L)

Run the natural design workflow with:

.. code-block:: bash

   python plifepred2.py --mode natural --design --input peptides.fasta --output natural_design.tsv

2. Modified peptide design
--------------------------

For each input peptide, PLifePred2 evaluates all 32 binary combinations of
the five modification flags (including the all-zero combination). The amino-
acid sequence itself is not mutated in this workflow.

.. code-block:: bash

   python plifepred2.py --mode modified  --design --input peptides.fasta --output modification_design.tsv

When ``--design`` is used with ``--mode modified``, any supplied
``--modifications``/``--mods`` value is ignored because all 32 combinations
are generated automatically.

Design sequence-length range
----------------------------

Design mode accepts peptide sequences containing **15 to 50 amino acids**
(inclusive). Each sequence is checked before design generation. This length
restriction applies to design mode; the standalone prediction code does not
apply this same explicit 15--50 residue check.


Input File Formats and Validation
=================================

Sequence input
--------------

Sequence modes and design mode accept FASTA files or plain-text files with
one peptide sequence per non-empty line.

FASTA example:

.. code-block:: text

   >Peptide_1
   ACDEFGHIKLMNPQRSTVWY
   >Peptide_2
   KLVFFAE

Plain-text example:

.. code-block:: text

   ACDEFGHIKLMNPQRSTVWY
   KLVFFAE

Only the 20 standard amino-acid symbols are accepted:

.. code-block:: text

   ACDEFGHIKLMNPQRSTVWY

Sequences are converted to uppercase before validation. Ambiguous or
non-standard residue symbols (for example, ``X``, ``B``, ``Z``, ``U``, and
``O``) are not accepted by the current input parser.

Multiple sequence files can be passed in one command:

.. code-block:: bash

   python plifepred2.py --mode natural --input group_1.fasta group_2.fasta additional_sequences.txt --output combined_predictions.tsv

FASTA record IDs are taken from the first whitespace-delimited token in each
header. Duplicate IDs across input files are made unique automatically.

SMILES input
------------

Use one SMILES string per non-empty line. An optional second whitespace-
separated token is interpreted as the molecule ID. Do not add extra fields to
a line.

PDB input
---------

Supply one or more PDB files. Each file is treated as one molecular input.
The program does not split multiple ``MODEL`` records inside a single PDB file
into separate predictions.


Output Files
============

All prediction results are written as tab-separated values (TSV), regardless
of the output filename extension. The default output name is
``plifepred2_predictions.tsv``.

Sequence prediction output
--------------------------

The sequence prediction table contains:

+----------------------+-------------------------------------------------+
| Column               | Description                                     |
+======================+=================================================+
| ``ID``               | Input FASTA/plain-text record identifier.       |
+----------------------+-------------------------------------------------+
| ``Sequence``         | Validated amino-acid sequence.                  |
+----------------------+-------------------------------------------------+
| ``Prediction``       | Predicted half-life on the model's              |
|                      | inverse-transformed target scale.               |
+----------------------+-------------------------------------------------+
| ``InputType``        | ``natural`` or ``modified``.                    |
+----------------------+-------------------------------------------------+
| ``Modifications``    | Selected modification flags, or ``none``.       |
+----------------------+-------------------------------------------------+
| ``ModificationFlags``| Binary values of ``d``, ``ct``, ``nt``,         |
|                      | ``cyc``, and ``ptm``.                           |
+----------------------+-------------------------------------------------+

SMILES and structure prediction output
--------------------------------------

The SMILES and structure prediction tables contain:

+----------------+-------------------------------------------------+
| Column         | Description                                     |
+================+=================================================+
| ``ID``         | Molecule ID or PDB filename stem.               |
+----------------+-------------------------------------------------+
| ``InputType``  | ``smiles`` or ``structure``.                    |
+----------------+-------------------------------------------------+
| ``SMILES``     | Canonical SMILES used for descriptor creation.  |
+----------------+-------------------------------------------------+
| ``Prediction`` | Predicted half-life on the model's              |
|                | inverse-transformed target scale.               |
+----------------+-------------------------------------------------+

Design output
-------------

Design results contain the following fields:

+------------------------------+-------------------------------------------+
| Column                       | Description                               |
+==============================+===========================================+
| ``Rank``                     | Rank within the corresponding input       |
|                              | peptide; highest prediction is rank 1.    |
+------------------------------+-------------------------------------------+
| ``Input_ID``                 | Identifier of the original peptide.       |
+------------------------------+-------------------------------------------+
| ``Variant``                  | Generated variant identifier.             |
+------------------------------+-------------------------------------------+
| ``Mutation``                 | Single substitution or wild-type label.   |
+------------------------------+-------------------------------------------+
| ``Position``                 | One-based residue position, if mutated.   |
+------------------------------+-------------------------------------------+
| ``Original_AA``              | Original residue, if mutated.             |
+------------------------------+-------------------------------------------+
| ``New_AA``                   | Substituted residue, if mutated.          |
+------------------------------+-------------------------------------------+
| ``Sequence``                 | Sequence used for the prediction.         |
+------------------------------+-------------------------------------------+
| ``ModificationCombination``  | Five-character binary flag combination.   |
+------------------------------+-------------------------------------------+
| ``d``, ``ct``, ``nt``,       | Individual binary modification flags.     |
| ``cyc``, ``ptm``             |                                           |
+------------------------------+-------------------------------------------+
| ``Prediction``               | Predicted half-life on the model's        |
|                              | inverse-transformed target scale.         |
+------------------------------+-------------------------------------------+
| ``Mode``                     | ``design_natural`` or ``design_modified``.|
+------------------------------+-------------------------------------------+

For each input peptide, design variants are sorted from the highest to the
lowest predicted value. The exact physical unit of ``Prediction`` is the unit
used for the model's training target; confirm that unit from the associated
model documentation before interpreting or reporting values.


Intermediate Files and Reproducibility
=======================================

By default, intermediate files are written to a temporary directory and
removed when the program finishes. To retain them, provide ``--workdir``:

.. code-block:: bash

   python plifepred2.py \
       --mode smiles \
       --input molecules.smi \
       --output smiles_predictions.tsv \
       --workdir run_intermediates

The retained files include generated descriptor tables and temporary inputs
used by the relevant workflow. In SMILES and structure modes, the complete
Mordred descriptor table is also saved in the working directory.

The sequence workflows use the Pfeature descriptor groups configured in the
script. The SMILES and structure workflows use the bundled Mordred-community
implementation. The structure model applies its serialized preprocessing
pipeline to the Mordred descriptor matrix. Keep the feature lists, model
files, compatibility module, and vendored Mordred source together with the
script to preserve the expected inference workflow.




Citation
========

If PLifePred2 is used in a publication, please cite the associated PLifePred2
paper or manuscript. Add the full bibliographic reference and DOI here when
the publication details are finalized.


License
=======

PLifePred2 is distributed under the GNU General Public License, version 3
(GPL-3.0). Include the corresponding ``LICENSE`` file in the source
repository and verify that the redistribution terms of bundled models,
Pfeature, and vendored third-party components are compatible with the
licenses under which they were obtained.


Support
=======

For bug reports, questions, and feature requests, please use the project's
GitHub issue tracker. Add the final repository URL and maintainer contact
information to this section before publishing the distribution.
