import cirq
import numpy as np
import pytest

import recirq.contextuality.magic_square_game as msg


@pytest.mark.parametrize(
    "game",
    ["infer_3rd", "measure_3rd_classical_multiplication", "measure_3rd_quantum_multiplication"],
)
@pytest.mark.parametrize("add_dd", [True, False])
def test_run_contextuality_experiment(game: msg.GameType, add_dd: bool) -> None:
    """Test that Alice and Bob win 100% of the time with a noiseless simulator."""

    sampler = cirq.Simulator()
    alice_qubits = cirq.GridQubit.rect(1, 4, 0, 0)  # test with 2 measure qubits
    bob_qubits = cirq.GridQubit.rect(1, 4, 1, 0)
    result = msg.run_contextuality_experiment(
        sampler,
        alice_qubits,
        bob_qubits,
        game=game,
        add_dd=add_dd,
        sub_case="square_1",
        repetitions=10,
    )
    assert np.all(result.get_agree_given_multiply_matrix(game) == np.ones((3, 3)))


@pytest.mark.parametrize(
    "game",
    ["infer_3rd", "measure_3rd_classical_multiplication", "measure_3rd_quantum_multiplication"],
)
@pytest.mark.parametrize("add_dd", [True, False])
def test_invalid_input_raises(game: msg.GameType, add_dd: bool):
    sampler = cirq.Simulator()
    alice_qubits = cirq.GridQubit.rect(1, 4, 0, 0)  # test with 2 measure qubits
    bob_qubits = cirq.GridQubit.rect(1, 4, 1, 0)
    with pytest.raises(ValueError, match="infer_3rd if you sub_case = only_two_qubits"):
        _ = msg.run_contextuality_experiment(
            sampler,
            alice_qubits,
            bob_qubits,
            game=game,
            add_dd=add_dd,
            sub_case="only_two_qubits",
            repetitions=10,
        )
