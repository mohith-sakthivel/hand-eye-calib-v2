import torch
import theseus as th

from typing import List, Optional, Tuple
from theseus.geometry import compose, between


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


def build_BA_Layer(
    E_T_C_values: torch.Tensor,
    C_T_C_values: torch.Tensor,
    E_T_E_values: torch.Tensor,
    edge_index: torch.Tensor,
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
        cost_weight.to(E_T_C_values.device)

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
    layer.to(E_T_C_values.device)

    return layer
