
#include <cstdlib>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/numbers.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_common.h>
#include <deal.II/numerics/vector_tools_constraints.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <set>

#include <deal.II/grid/grid_refinement.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/base/thread_management.h>

#include <deal.II/base/timer.h>
#include <string>
#include <system_error>
#include <vector>


dealii::UpdateFlags POISSON_VOLUME_FLAGS = 
    dealii::update_gradients |
    dealii::update_JxW_values |
    dealii::update_quadrature_points;

template<int dim>
void assemble_local_poisson_volume_matrix(
    const dealii::FEValues<dim> &fe_values,
    const dealii::Function<dim> &permittivity,
    dealii::FullMatrix<double> &local_mat
) {
    const auto flags = fe_values.get_update_flags();
    Assert((flags & POISSON_VOLUME_FLAGS) == POISSON_VOLUME_FLAGS,
       dealii::ExcMessage("assemble_local_poisson_volume_matrix: FEValues missing required update flags."));

    for (const uint q : fe_values.quadrature_point_indices()) {
        const dealii::Point<dim> &x_q = fe_values.quadrature_point(q);
        for (uint i = 0; i < fe_values.get_fe().dofs_per_cell; i++) {
            for (uint j = 0; j < fe_values.get_fe().dofs_per_cell; j++) {
                local_mat(i, j) += fe_values.shape_grad(i, q) 
                    * permittivity.value(x_q) 
                    * fe_values.shape_grad(j, q) 
                    * fe_values.JxW(q);
            }
        }
    }
}

dealii::UpdateFlags POISSON_BOUNDARY_FLAGS = 
    dealii::update_values |
    dealii::update_normal_vectors |
    dealii::update_gradients |
    dealii::update_JxW_values  |
    dealii::update_quadrature_points;

template<int dim>
void assemble_local_poisson_boundary_matrix(
    const dealii::FEFaceValues<dim> &fe_face_values,
    const dealii::Function<dim> &permittivity,
    dealii::FullMatrix<double> &local_mat
) {
    const auto flags = fe_face_values.get_update_flags();
    Assert((flags & POISSON_BOUNDARY_FLAGS) == POISSON_BOUNDARY_FLAGS,
       dealii::ExcMessage("assemble_local_poisson_boundary_matrix: FEValues missing required update flags."));

    for (const uint q : fe_face_values.quadrature_point_indices()) {
        const dealii::Point<dim> &x_q = fe_face_values.quadrature_point(q);
        const double eps = permittivity.value(x_q);

        for (uint i = 0; i < fe_face_values.get_fe().dofs_per_cell; ++i) {
            for (uint j = 0; j < fe_face_values.get_fe().dofs_per_cell; ++j) {
                const dealii::Tensor<1,dim>& normal = fe_face_values.normal_vector(q);
                local_mat(i, j) -= fe_face_values.shape_value(i, q) 
                    * normal * eps * fe_face_values.shape_grad(j, q) 
                    * fe_face_values.JxW(q);
            }
        }
    }
}

template<int dim>
void assemble_poisson_system(
    const dealii::DoFHandler<dim>& dof_handler,
    const dealii::AffineConstraints<double>& constraints,
    const dealii::Function<dim>& permittivity,
    dealii::SparseMatrix<double>& system_matrix,
    dealii::Vector<double>& system_rhs
) {
    auto& fe = dof_handler.get_fe();
    
    dealii::QGauss<dim> quadrature{fe.degree + 1};
    dealii::QGauss<dim-1> face_quadrature{fe.degree + 1};

    dealii::FEValues<dim> fe_values{fe, quadrature, POISSON_VOLUME_FLAGS };
    dealii::FEFaceValues<dim> fe_face_values{fe, face_quadrature, POISSON_BOUNDARY_FLAGS };

    dealii::FullMatrix<double> local_mat(fe.dofs_per_cell, fe.dofs_per_cell);
    dealii::Vector<double> cell_rhs(fe.dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof(fe.dofs_per_cell);

    for (const auto& cell : dof_handler.active_cell_iterators()) {
        local_mat = 0;
        cell_rhs = 0;

        fe_values.reinit(cell);
        assemble_local_poisson_volume_matrix(fe_values, permittivity, local_mat);

        for (uint face = 0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
            if (!cell->face(face)->at_boundary()) continue;
            fe_face_values.reinit(cell, face);
            assemble_local_poisson_boundary_matrix(fe_face_values, permittivity, local_mat);
        }

        cell->get_dof_indices(local_dof);
        constraints.distribute_local_to_global(local_mat, cell_rhs, local_dof, system_matrix, system_rhs);
    }
}

template <typename MatrixType, typename VectorType>
void solve_cg(const MatrixType& system_matrix, VectorType& solution, const VectorType& rhs) {
    dealii::SolverControl solver_control(1000, 1e-6 * rhs.l2_norm());
    dealii::PreconditionSSOR<MatrixType> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);

    dealii::SolverCG<VectorType> solver(solver_control);
    solver.solve(system_matrix, solution, rhs, dealii::PreconditionIdentity());
    std::cout << solver_control.last_step() << " CG iterations needed to converge.\n";
}

template <int dim>
dealii::Vector<double> solve_poisson_system(
    const dealii::DoFHandler<dim>& dof_handler,
    const dealii::Function<dim>& permittivity,
    const dealii::AffineConstraints<double>& user_constraints
) {

    dealii::AffineConstraints<double> constraints{user_constraints};
    dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();

    dealii::DynamicSparsityPattern dsp(dof_handler.n_dofs());
    dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    dealii::SparsityPattern sp;
    sp.copy_from(dsp);

    dealii::Vector<double> solution;
    dealii::Vector<double> system_rhs;
    dealii::SparseMatrix<double> system_matrix;

    system_matrix.reinit(sp);
    system_rhs.reinit(dof_handler.n_dofs());
    solution.reinit(dof_handler.n_dofs());

    assemble_poisson_system(dof_handler, constraints, permittivity, system_matrix, system_rhs);
    solve_cg(system_matrix, solution, system_rhs);
    constraints.distribute(solution);

    return solution;
}

template<int dim>
double smallest_cell_size(const dealii::Triangulation<dim>& triangulation) {
    double h_min = std::numeric_limits<double>::max();
    for (const auto &cell : triangulation.active_cell_iterators())
        h_min = std::min(h_min, cell->minimum_vertex_distance());
    return h_min;
}

template<int dim>
double get_l2_error(
    const dealii::DoFHandler<dim>& dof_handler,
    const dealii::Vector<double>& solution,
    const dealii::Function<dim>& ex_solution
) {
    dealii::Vector<double> difference_per_cell(dof_handler.get_triangulation().n_active_cells());
    dealii::VectorTools::integrate_difference(
        dof_handler,
        solution,
        ex_solution,
        difference_per_cell,
        dealii::QGauss<2>(dof_handler.get_fe().degree + 1),
        dealii::VectorTools::L2_norm
    );

    return difference_per_cell.l2_norm();
}


template<int dim>
void write_out_solution(
    const dealii::DoFHandler<dim>& dof_handler,
    const dealii::Vector<double>& solution,
    std::string file) 
{
    dealii::DataOut<2> data_out;
    data_out.attach_dof_handler(dof_handler);
    std::vector<std::string> solution_names;

    solution_names.emplace_back("potential");
    data_out.add_data_vector(solution, solution_names);
    data_out.build_patches();

    std::ofstream output(file);
    data_out.write_vtu(output);
}

struct RadialCapacitor{
    const double r0;
    const double r1;
    const double r2;

    const double voltage0;
    const double voltage1;

    const double epsilon0_1;
    const double epsilon1_2;
};

template <int dim>
class PermittivityFunction : public dealii::Function<dim> {
public:
    const RadialCapacitor capacitor;
    PermittivityFunction(const RadialCapacitor capacitor) : capacitor(capacitor) {}

    double value(const dealii::Point<dim> &p, const unsigned int = 0) const override {
        return p.norm() < capacitor.r1 ? capacitor.epsilon0_1 : capacitor.epsilon1_2;
    }
};

class Exact2DPotentialSolution : public dealii::Function<2> {
private:
    // phi(r ∈ [r0, r1]) = solution[0] ln r + solution[2]
    // phi(r ∈ [r1, r2]) = solution[2] ln r + solution[3]
    dealii::Vector<double> consts; 

public:
    const RadialCapacitor capacitor;
    Exact2DPotentialSolution(const RadialCapacitor capacitor) 
        : capacitor(capacitor) 
    {
        const double rhs_vec[4] = { capacitor.voltage0, capacitor.voltage1, 0, 0 };
        const double system_mat[4][4] = {
            { std::log(capacitor.r0), 1.0, 0.0, 0.0 },
            { 0.0, 0.0, std::log(capacitor.r2), 1.0 },
            { std::log(capacitor.r1), 1.0, -std::log(capacitor.r1), -1.0 },
            { capacitor.epsilon0_1, 0.0, -capacitor.epsilon1_2, 0.0 }
        };

        dealii::FullMatrix<double> system(4, 4); // system matrix 
        dealii::Vector<double> rhs(4);
        consts.reinit(4);

        for (uint i=0; i < 4; i++) {
            rhs[i] = rhs_vec[i];
            for (uint j = 0; j < 4; j++) {
                system(i,j) = system_mat[i][j];
            }
        }

        system.gauss_jordan();             // in-place Gauss-Jordan invert
        system.vmult(consts, rhs);         // x = A^{-1} * b
    }

    double value(const dealii::Point<2> &p, const unsigned int = 0) const override {
        double r = p.norm();
        return (r < capacitor.r1)
            ? consts[0]*std::log(r) + consts[1] 
            : consts[2]*std::log(r) + consts[3];
    }
};

int main() {

    // problem definition

    const RadialCapacitor capacitor{0.5, 0.75, 1., 0., 1., 1., 10.}; // r0, r1, r2, U0, U2, eps0_1, eps1_2
    const Exact2DPotentialSolution ex_solution(capacitor); 
    const PermittivityFunction<2> permittivity{capacitor};

    // Create triangulation

    dealii::Triangulation<2> triangulation;
    const dealii::Point<2> center(0, 0);
    dealii::GridGenerator::hyper_shell( triangulation, center, capacitor.r0, capacitor.r2, 10, true);
    triangulation.refine_global(2);

    // boundary ids assigned in dealii::GridGenerator::hyper_shell
    const dealii::types::boundary_id inner_id = 0, outer_id = 1; 

    // solve

    dealii::FE_Q<2> fe{1};
    dealii::DoFHandler<2> dof_handler{triangulation};
    dof_handler.distribute_dofs(fe);

    std::ofstream error_file("l2_errors.txt");

    for (int i = 0; i < 9; i++) {

        dealii::AffineConstraints<double> constraints;
        dealii::VectorTools::interpolate_boundary_values(dof_handler, inner_id, dealii::Functions::ConstantFunction<2>(capacitor.voltage0), constraints); 
        dealii::VectorTools::interpolate_boundary_values(dof_handler, outer_id, dealii::Functions::ConstantFunction<2>(capacitor.voltage1), constraints); 

        dealii::Vector<double> solution = solve_poisson_system(dof_handler, permittivity, constraints);
        write_out_solution(dof_handler, solution, "solutions/solution" + std::to_string(i) + ".vtu");

        // l2 error

        double l2_error = get_l2_error(dof_handler, solution, ex_solution);
        std::cout << "smallest cell size: " << smallest_cell_size(triangulation) << " l2: " << l2_error << "\n";
        error_file << smallest_cell_size(triangulation) << " " << l2_error << "\n";
        error_file.flush();

        // refinement

        dealii::Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
        dealii::KellyErrorEstimator<2>::estimate( dof_handler, dealii::QGauss<1>(fe.degree + 1), {}, solution, estimated_error_per_cell);
        dealii::GridRefinement::refine_and_coarsen_fixed_number(triangulation, estimated_error_per_cell, 0.3, 0.03);

        //triangulation.refine_global(1);
        triangulation.execute_coarsening_and_refinement();
        dof_handler.distribute_dofs(fe);
    }

    error_file.close();
}
