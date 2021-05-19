# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Adds q^T * w to separate register for non-negative integer weights w."""

import logging

import warnings
import numpy as np

from qiskit.circuit.library.arithmetic import WeightedAdder
from qiskit.aqua import AquaError

logger = logging.getLogger(__name__)


from abc import ABC, abstractmethod
from qiskit import QuantumCircuit
from qiskit.aqua.utils.controlled_circuit import get_controlled_circuit


class CircuitFactory(ABC):

    """ Base class for CircuitFactories """

    def __init__(self, num_target_qubits: int) -> None:
        self._num_target_qubits = num_target_qubits
        pass

    @property
    def num_target_qubits(self):
        """ Returns the number of target qubits """
        return self._num_target_qubits

    def required_ancillas(self):
        """ returns required ancillas """
        return 0

    def required_ancillas_controlled(self):
        """ returns required ancillas controlled """
        return self.required_ancillas()

    def get_num_qubits(self):
        """ returns number of qubits """
        return self._num_target_qubits + self.required_ancillas()

    def get_num_qubits_controlled(self):
        """ returns number of qubits controlled """
        return self._num_target_qubits + self.required_ancillas_controlled()

    @abstractmethod
    def build(self, qc, q, q_ancillas=None, params=None):
        """ Adds corresponding sub-circuit to given circuit

        Args:
            qc (QuantumCircuit): quantum circuit
            q (list): list of qubits (has to be same length as self._num_qubits)
            q_ancillas (list): list of ancilla qubits (or None if none needed)
            params (list): parameters for circuit
        """
        raise NotImplementedError()

    def build_inverse(self, qc, q, q_ancillas=None):
        """ Adds inverse of corresponding sub-circuit to given circuit

        Args:
            qc (QuantumCircuit): quantum circuit
            q (list): list of qubits (has to be same length as self._num_qubits)
            q_ancillas (list): list of ancilla qubits (or None if none needed)
        """
        qc_ = QuantumCircuit(*qc.qregs)

        self.build(qc_, q, q_ancillas)
        qc.extend(qc_.inverse())

    def build_controlled(self, qc, q, q_control, q_ancillas=None, use_basis_gates=True):
        """ Adds corresponding controlled sub-circuit to given circuit

        Args:
            qc (QuantumCircuit): quantum circuit
            q (list): list of qubits (has to be same length as self._num_qubits)
            q_control (Qubit): control qubit
            q_ancillas (list): list of ancilla qubits (or None if none needed)
            use_basis_gates (bool): use basis gates for expansion of controlled circuit
        """
        uncontrolled_circuit = QuantumCircuit(*qc.qregs)
        self.build(uncontrolled_circuit, q, q_ancillas)

        controlled_circuit = get_controlled_circuit(uncontrolled_circuit,
                                                    q_control, use_basis_gates=use_basis_gates)
        qc.extend(controlled_circuit)

    def build_controlled_inverse(self, qc, q, q_control, q_ancillas=None, use_basis_gates=True):
        """ Adds controlled inverse of corresponding sub-circuit to given circuit

        Args:
            qc (QuantumCircuit): quantum circuit
            q (list): list of qubits (has to be same length as self._num_qubits)
            q_control (Qubit): control qubit
            q_ancillas (list): list of ancilla qubits (or None if none needed)
            use_basis_gates (bool): use basis gates for expansion of controlled circuit
        """
        qc_ = QuantumCircuit(*qc.qregs)

        self.build_controlled(qc_, q, q_control, q_ancillas, use_basis_gates)
        qc.extend(qc_.inverse())

    def build_power(self, qc, q, power, q_ancillas=None):
        """ Adds power of corresponding circuit.
            May be overridden if a more efficient implementation is possible """
        for _ in range(power):
            self.build(qc, q, q_ancillas)

    def build_inverse_power(self, qc, q, power, q_ancillas=None):
        """ Adds inverse power of corresponding circuit.
            May be overridden if a more efficient implementation is possible """
        for _ in range(power):
            self.build_inverse(qc, q, q_ancillas)

    def build_controlled_power(self, qc, q, q_control, power,
                               q_ancillas=None, use_basis_gates=True):
        """ Adds controlled power of corresponding circuit.
            May be overridden if a more efficient implementation is possible """
        for _ in range(power):
            self.build_controlled(qc, q, q_control, q_ancillas, use_basis_gates)

    def build_controlled_inverse_power(self, qc, q, q_control, power,
                                       q_ancillas=None, use_basis_gates=True):
        """ Adds controlled, inverse, power of corresponding circuit.
            May be overridden if a more efficient implementation is possible """
        for _ in range(power):
            self.build_controlled_inverse(qc, q, q_control, q_ancillas, use_basis_gates)

class WeightedSumOperator(CircuitFactory):
    """Adds q^T * w to separate register for non-negative integer weights w."""

    def __init__(self, num_state_qubits, weights, i_state=None, i_sum=None):
        """Computes the weighted sum controlled by state qubits

        Args:
            num_state_qubits (int): number of state qubits
            weights (Union(list, numpy.ndarray)): weights per state qubits
            i_state (Optional(Union(list, numpy.ndarray))): indices of state qubits,
                                    set to range(num_state_qubits) if None
            i_sum (Optional(int)): indices of target qubits (that represent the resulting sum),
                set to range(num_state_qubits, num_state_qubits + req_num_sum_qubits) if None
        Raises:
            AquaError: invalid input
        """
        self._weights = weights

        # check weights
        for i, w in enumerate(weights):
            if not np.isclose(w, np.round(w)):
                logger.warning('Non-integer weights are rounded to '
                               'the nearest integer! (%s, %s).', i, w)

        self._num_state_qubits = num_state_qubits
        self._num_sum_qubits = self.get_required_sum_qubits(weights)
        self._num_carry_qubits = self.num_sum_qubits - 1

        num_target_qubits = num_state_qubits + self.num_sum_qubits
        super().__init__(num_target_qubits)

        if i_state is None:
            self.i_state = list(range(num_state_qubits))
        else:
            self.i_state = i_state
        if i_sum is None:
            self.i_sum = \
                list(range(max(self.i_state) + 1, max(self.i_state) + self.num_sum_qubits + 1))
        else:
            if len(i_sum) == self.get_required_sum_qubits(weights):
                self.i_sum = i_sum
            else:
                raise AquaError('Invalid number of sum qubits {}! Required {}'.format(
                    len(i_sum), self.get_required_sum_qubits(weights)
                ))

    @staticmethod
    def get_required_sum_qubits(weights):
        """ get required sum qubits """
        return int(np.floor(np.log2(sum(weights))) + 1)

    @property
    def weights(self):
        """ returns weights """
        return self._weights

    @property
    def num_state_qubits(self):
        """ returns num state qubits """
        return self._num_state_qubits

    @property
    def num_sum_qubits(self):
        """ returns num sum qubits """
        return self._num_sum_qubits

    @property
    def num_carry_qubits(self):
        """ returns num carry qubits """
        return self._num_carry_qubits

    def required_ancillas(self):
        """ required ancillas """
        if self.num_sum_qubits > 2:
            # includes one ancilla qubit for 3-controlled not gates
            # TODO: validate when the +1 is needed and make a case distinction
            return self.num_carry_qubits + 1
        else:
            return self.num_carry_qubits

    def required_ancillas_controlled(self):
        """ returns required ancillas controlled """
        return self.required_ancillas()

    def build(self, qc, q, q_ancillas=None, params=None):
        instr = WeightedAdder(num_state_qubits=self.num_state_qubits,
                              weights=self.weights).to_instruction()
        qr = [q[i] for i in self.i_state + self.i_sum]
        if q_ancillas:
            qr += q_ancillas[:self.required_ancillas()]  # pylint:disable=unnecessary-comprehension
        qc.append(instr, qr)
