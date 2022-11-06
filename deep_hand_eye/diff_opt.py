from typing import Optional, List

import torch
import theseus as th
from theseus.geometry import compose, between


class PoseConsistency(th.CostFunction):

    def __init__(
        self,
        C1_T_C2: th.SE3,
        E1_T_E2: th.SE3,
        E_T_C: th.SE3,
        weight: Optional[th.CostWeight] = None,
        name: Optional[str] = None
    ):
        if weight is None:
            weight = th.ScaleCostWeight(torch.tensor(1.0).to(dtype=E_T_C.dtype))

        super().__init__(cost_weight=weight, name=name)

        self.C1_T_C2 = C1_T_C2
        self.E1_T_E2 = E1_T_E2
        self.E_T_C = E_T_C

        self.register_optim_vars(["E_T_C"])
        self.register_aux_vars(["C1_T_C2", "E1_T_E2"])

    def error(self) -> torch.Tensor:
        E1_T_E2_est = compose(self.E_T_C, compose(self.C1_T_C2, self.E_T_C.inverse()))
        error = E1_T_E2_est.local(self.E1_T_E2)
        return error

    def dim(self) -> int:
        return self.E_T_C.dof()

    def _copy_impl(self, new_name: Optional[str] = None) -> "PoseConsistency":
        return PoseConsistency(
            self.C1_T_C2.copy(),
            self.E1_T_E2.copy(),
            self.E_T_C.copy(),
            weight=self.weight.copy(),
            name=new_name
        )


class DifferentiableOptimizer:

    def __init__(self):
        self.objective = th.Objective(dtype=torch.float32)

        rel_ee_motion = th.SE3(name="rel_ee_motion", dtype=torch.float32)
        rel_cam_motion = th.SE3(name="re_cam_motion", dtype=torch.float32)
        hand_eye = th.SE3(name="hand_eye", dtype=torch.float32)

        self.cost = PoseConsistency(rel_cam_motion, rel_ee_motion, hand_eye)
        self.objective.add(self.cost)

        self.optimizer = th.GaussNewton(
            objective=self.objective,
            max_iterations=10,
            step_size=1,
            linearization_cls=th.SparseLinearization,
            linear_solver_cls=th.CholmodSparseSolver,
            vectorize=True
        )

        self.diff_opt_layer = th.TheseusLayer(self.optimizer)

    def enforce(self, rel_cam_motion, rel_ee_motion, hand_eye):
        rel_cam_motion = th.SE3(x_y_z_quaternion=rel_cam_motion)
        rel_ee_motion = th.SE3(x_y_z_quaternion=rel_ee_motion)
        hand_eye = th.SE3(x_y_z_quaternion=hand_eye)

        results, _ = self.diff_opt_layer.forward({
            "rel_ee_motion": rel_ee_motion,
            "rel_cam_motion": rel_cam_motion,
            "hand_eye": hand_eye
        })

        return results["hand_eye"]
