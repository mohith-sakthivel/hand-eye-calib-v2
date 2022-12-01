import torch
import pickle
import numpy as np
import theseus as th

from typing import List, Optional, Tuple
from theseus.geometry import compose, local, between
from deep_hand_eye.pose_utils import quaternion_angular_error, homo_to_quat


DEVICE = "cuda"


def to_tensor(array: np.ndarray, device: str = DEVICE) -> torch.Tensor:
    return torch.tensor(array).to(device)


class PoseConsistency(th.CostFunction):
    def __init__(
        self,
        CA_T_CB: th.SE3,
        EA_T_EB: th.SE3,
        E_T_C: th.SE3,
        cost_weight: Optional[th.CostWeight] = None,
        name: Optional[str] = None,
    ):
        if cost_weight is None:
            cost_weight = th.ScaleCostWeight(torch.tensor(1.0).to(dtype=E_T_C.dtype))

        super().__init__(cost_weight=cost_weight, name=name)

        self.CA_T_CB = CA_T_CB
        self.EA_T_EB = EA_T_EB
        self.E_T_C = E_T_C

        self.register_optim_vars(["E_T_C"])
        self.register_aux_vars(["EA_T_EB", "CA_T_CB"])

    def error(self) -> torch.Tensor:
        EA_T_EB_est = compose(self.E_T_C, compose(self.CA_T_CB, self.E_T_C.inverse()))
        error = self.EA_T_EB.local(EA_T_EB_est)

        return error

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        log_jac = []

        # Method 1
        CA_T_CB_est = compose(self.E_T_C.inverse(), compose(self.EA_T_EB, self.E_T_C))
        error = between(self.CA_T_CB, CA_T_CB_est).log_map(jacobians=log_jac)

        dlog = log_jac[0]
        JList = [-dlog @ CA_T_CB_est.inverse().adjoint() + dlog,]

        # # Method 2
        # EA_T_EB_est = compose(self.E_T_C, compose(self.CA_T_CB, self.E_T_C.inverse()))
        # error = between(self.EA_T_EB, EA_T_EB_est).log_map(jacobians=log_jac)

        # dlog = log_jac[0]
        # JList = [(dlog @ EA_T_EB_est.inverse().adjoint() - dlog) @ self.E_T_C.adjoint(),]

        return JList, error

    def dim(self) -> int:
        return self.E_T_C.dof()

    def _copy_impl(self, new_name: Optional[str] = None) -> "PoseConsistency":
        return PoseConsistency(
            self.CA_T_CB.copy(),
            self.EA_T_EB.copy(),
            self.E_T_C.copy(),
            cost_weight=self.weight.copy(),
            name=new_name,
        )


def pose_consistency_error(optim_vars, aux_vars):
    E_T_C = optim_vars[0]
    E_T_E, C_T_C = aux_vars
    E_T_E_est = compose(E_T_C, compose(C_T_C, E_T_C.inverse()))
    error = local(E_T_E, E_T_E_est)

    return error


def build_BA_LAyer(
    E_T_C_values: torch.Tensor,
    C_T_C_values: torch.Tensor,
    E_T_E_values: torch.Tensor,
    edge_index: torch.Tensor,
    use_auto_diff: bool = False,
) -> th.TheseusLayer:

    objective = th.Objective()
    E_T_C = th.SE3(x_y_z_quaternion=E_T_C_values, name="E_T_C")

    for i in range(len(edge_index[0, 0])):
        edge_i, edge_j = edge_index[0, 0, i], edge_index[0, 1, i]
        C_T_C = th.SE3(
            x_y_z_quaternion=C_T_C_values[:, i], name=f"C{edge_i}_T_C{edge_j}"
        )
        E_T_E = th.SE3(
            x_y_z_quaternion=E_T_E_values[:, i], name=f"E{edge_i}_T_E{edge_j}"
        )

        cost_weight = th.ScaleCostWeight(1.0)
        cost_weight.to(DEVICE)

        if use_auto_diff:
            cost_fn = th.AutoDiffCostFunction(
                optim_vars=(E_T_C,),
                err_fn=pose_consistency_error,
                dim=6,
                cost_weight=cost_weight,
                aux_vars=(E_T_E, C_T_C),
                name=f"cost_{edge_i}_to_{edge_j}",
            )
        else:
            cost_fn = PoseConsistency(
                CA_T_CB=C_T_C,
                EA_T_EB=E_T_E,
                E_T_C=E_T_C,
                cost_weight=cost_weight,
                name=f"cost_{edge_i}_to_{edge_j}",
            )

        objective.add(cost_fn)

    optimizer = th.GaussNewton(
        objective=objective,
        max_iterations=10,
        step_size=1,
        linearization_cls=th.SparseLinearization,
        linear_solver_cls=th.CholmodSparseSolver,
        vectorize=True,
    )

    layer = th.TheseusLayer(optimizer)
    layer.to(DEVICE)

    return layer


def run():
    with open("dev/sample_data.pkl", "rb") as f:
        data = pickle.load(f)

    START_IDX = 0
    END_IDX = len(data["he_actual"])

    batch_data = {
        "E_T_C_pred": to_tensor(data["he_pred"][START_IDX:END_IDX], DEVICE),
        "C_T_C_pred": to_tensor(data["rel_cam_pred"][START_IDX:END_IDX], DEVICE),
        "E_T_E_true": to_tensor(data["rel_ee"][START_IDX:END_IDX], DEVICE),
        "edge_index": to_tensor(data["edge_idx"][START_IDX:END_IDX], DEVICE),
        "E_T_C_true": to_tensor(data["he_actual"][START_IDX:END_IDX], DEVICE),
        "C_T_C_true": to_tensor(data["rel_cam_actual"][START_IDX:END_IDX], DEVICE),
    }

    theseus_layer = build_BA_LAyer(
        E_T_C_values=batch_data["E_T_C_pred"],
        C_T_C_values=batch_data["C_T_C_pred"],
        E_T_E_values=batch_data["E_T_E_true"],
        edge_index=batch_data["edge_index"],
    )

    theseus_inputs = {"E_T_C": th.SE3(x_y_z_quaternion=batch_data["E_T_C_pred"])}
    for i in range(batch_data["edge_index"].shape[-1]):
        edge_i, edge_j = (
            batch_data["edge_index"][0, 0, i],
            batch_data["edge_index"][0, 1, i],
        )
        theseus_inputs[f"C{edge_i}_T_C{edge_j}"] = th.SE3(
            x_y_z_quaternion=batch_data["C_T_C_pred"][:, i]
        )
        theseus_inputs[f"E{edge_i}_T_E{edge_j}"] = th.SE3(
            x_y_z_quaternion=batch_data["E_T_E_true"][:, i]
        )

    with torch.no_grad():
        _, info = theseus_layer.forward(
            input_tensors=theseus_inputs,
            optimizer_kwargs={"track_best_solution": True, "verbose": True},
        )

    trans_optimized = info.best_solution["E_T_C"][:, :3, 3].cpu().numpy()
    trans_pred = batch_data["E_T_C_pred"][:, :3].cpu().numpy()
    trans_true = batch_data["E_T_C_true"][:, :3].cpu().numpy()

    quat_optimized = info.best_solution["E_T_C"].cpu().numpy()
    quat_optimized = np.array([homo_to_quat(T)[3:] for T in quat_optimized])
    quat_pred = batch_data["E_T_C_pred"][:, 3:].cpu().numpy()
    quat_true = batch_data["E_T_C_true"][:, 3:].cpu().numpy()

    trans_pred_error = np.linalg.norm(trans_pred - trans_true, axis=-1)
    trans_optimized_error = np.linalg.norm(trans_optimized - trans_true, axis=-1)
    quat_pred_error = quaternion_angular_error(quat_pred, quat_true)
    quat_optimized_error = quaternion_angular_error(quat_optimized, quat_true)

    print("Optimization Done... \n\n")
    print(f"Original Translation Error: \n\tMean: {np.mean(trans_pred_error): 0.4f} m \n\tMedian: {np.median(trans_pred_error): 0.4f} m")
    print(
        f"Translation Error after optimization: \n\tMean: {np.mean(trans_optimized_error): 0.4f} m \n\tMedian: {np.median(trans_optimized_error): 0.4f} m \n"
    )
    print(f"Original Rotation Error: \n\tMean: {np.mean(quat_pred_error):0.2f} deg \n\t Median: {np.median(quat_pred_error):0.2f} deg")
    print(
        f"Rotation Error after optimization: \n\tMean: {np.mean(quat_optimized_error): 0.2f} deg \n\tMedian: {np.median(quat_optimized_error): 0.2f} deg"
    )


if __name__ == "__main__":
    run()
