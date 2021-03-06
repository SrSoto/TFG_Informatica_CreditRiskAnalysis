# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Utilities (:mod:`qiskit.aqua.utils`)
========================================
Various utility functionality...

.. currentmodule:: qiskit.aqua.utils

Utilities
=========

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   tensorproduct
   random_unitary
   random_h2_body
   random_h1_body
   random_hermitian
   random_non_hermitian
   decimal_to_binary
   summarize_circuits
   get_subsystem_density_matrix
   get_subsystems_counts
   get_entangler_map
   validate_entangler_map
   get_feature_dimension
   get_num_classes
   split_dataset_to_data_and_labels
   map_label_to_class_name
   reduce_dim_to_via_pca
   optimize_svm
   CircuitFactory
   has_ibmq
   has_aer
   name_args

"""

from .circuit_factory import CircuitFactory

__all__ = [
    'CircuitFactory',
]
