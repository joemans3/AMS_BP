from functools import cache
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
from pydantic import BaseModel

from ..sample.flurophores.flurophore_schema import State, StateType
from ..sample.sim_sampleplane import FluorescentObject


class ErnoMsg(BaseModel):
    success: bool
    erno_time: Optional[float | None] = None
    erno_end_state: Optional[State | None] = None


class StateTransitionCalculator:
    def __init__(
        self,
        flurophoreobj: FluorescentObject,
        time_duration: int | float,
        current_global_time: int,
        laser_intensity_generator: Callable,
    ) -> None:
        self.flurophoreobj = flurophoreobj
        self.time_duration = time_duration  # seconds
        self.current_global_time = current_global_time  # ms (oversample motion time)
        self.laser_intensity_generator = laser_intensity_generator
        self.fluorescent_state_history = {}  # {fluorescent.state.name : [delta time (seconds), laser_intensites], ...}
        self.current_global_time_s = self.current_global_time * 1e-3

    def __call__(
        self,
    ) -> Tuple[dict, State, ErnoMsg]:
        state, erno = self.MCMC()
        return self.fluorescent_state_history, state, erno

    def _initialize_state_hist(self, time_pos: int, time_laser: float) -> dict:
        laser_intensities = self.laser_intensity_generator(
            florPos=self.flurophoreobj.position_history[time_pos],
            time=time_laser,
        )
        for i in self.flurophoreobj.fluorophore.states.values():
            if i.state_type == StateType.FLUORESCENT:
                self.fluorescent_state_history[i.name] = [0, laser_intensities]
        return laser_intensities

    def _get_intensities(self, time_pos: int, time_laser: float) -> dict:
        laser_intensities = self.laser_intensity_generator(
            florPos=self.flurophoreobj.position_history[time_pos],
            time=time_laser,
        )
        return laser_intensities

    def MCMC(self) -> Tuple[State, ErnoMsg]:
        time = 0
        transitions = self.flurophoreobj.state_history[self.current_global_time][2]
        if not transitions:
            self.fluorescent_state_history[
                self.flurophoreobj.fluorophore.states[
                    self.flurophoreobj.state_history[self.current_global_time][0].name
                ]
            ][0] += self.time_duration
            return self.flurophoreobj.fluorophore.states[
                self.flurophoreobj.state_history[self.current_global_time][0].name
            ], ErnoMsg(success=True)
        final_state_name = transitions[0].from_state
        laser_intensities = self._initialize_state_hist(
            self.current_global_time, time + self.current_global_time_s
        )
        while time < self.time_duration:
            laser_intensities = self._get_intensities(
                self.current_global_time, self.current_global_time_s + time
            )
            stateTransitionMatrixR = [
                sum(
                    state_transitions.rate()(laser["wavelength"], laser["intensity"])
                    for laser in laser_intensities.values()
                )
                for state_transitions in transitions
            ]  # 1/s
            if not stateTransitionMatrixR:
                if (
                    self.flurophoreobj.fluorophore.states[final_state_name].state_type
                    == StateType.FLUORESCENT
                ):
                    self.fluorescent_state_history[
                        self.flurophoreobj.fluorophore.states[final_state_name].name
                    ][0] += self.time_duration
                break
            if sum(stateTransitionMatrixR) == 0:
                if (
                    self.flurophoreobj.fluorophore.states[final_state_name].state_type
                    == StateType.FLUORESCENT
                ):
                    self.fluorescent_state_history[
                        self.flurophoreobj.fluorophore.states[final_state_name].name
                    ][0] += self.time_duration
                break

            # print(final_state_name)
            new_time, state_indx = ssa_step(
                stateTransitionMatrixR
            )  # seconds, index on transitions

            state_name = transitions[state_indx].to_state
            if new_time > self.time_duration:
                erno_time = new_time - time
                new_time = self.time_duration - time
                erno = ErnoMsg(
                    success=False,
                    erno_time=erno_time,
                    erno_end_state=self.flurophoreobj.fluorophore.states[state_name],
                )
                if (
                    self.flurophoreobj.fluorophore.states[final_state_name].state_type
                    == StateType.FLUORESCENT
                ):
                    # print("I glow inside")
                    self.fluorescent_state_history[
                        self.flurophoreobj.fluorophore.states[final_state_name].name
                    ][0] += new_time
                return self.flurophoreobj.fluorophore.states[final_state_name], erno

            # print(new_time)

            if (
                self.flurophoreobj.fluorophore.states[final_state_name].state_type
                == StateType.FLUORESCENT
            ):
                # print("I glow")
                self.fluorescent_state_history[
                    self.flurophoreobj.fluorophore.states[final_state_name].name
                ][0] += new_time
            final_state_name = state_name
            transitions = self._find_transitions(state_name)
            time += new_time
        # find state
        return self.flurophoreobj.fluorophore.states[final_state_name], ErnoMsg(
            success=True
        )

    @cache
    def _find_transitions(self, statename: str) -> list:
        return [
            stateTrans
            for stateTrans in self.flurophoreobj.fluorophore.transitions.values()
            if stateTrans.from_state == statename
        ]


def ssa_step(reaction_rates: Sequence[float | int]) -> tuple[float, int]:
    """
    Perform one step of the SSA simulation.

    Parameters:
    - reaction_rates: List of reaction rates [k1, k2, ...]

    Returns:
    - dt: Time step to advance
    - next_event: Index of the next reaction (0-based)
    """
    # Calculate propensities
    propensities = np.array(reaction_rates)
    total_propensity = np.sum(propensities)

    if total_propensity == 0:
        raise ValueError("Total propensity is zero; no reactions can occur.")

    # Draw two random numbers
    r1, r2 = np.random.uniform(0, 1, size=2)

    # Compute time step
    dt = -np.log(r1) / total_propensity

    # Determine the next reaction
    cumulative_propensities = np.cumsum(propensities)
    threshold = r2 * total_propensity
    next_event = np.searchsorted(cumulative_propensities, threshold)

    return dt, next_event
