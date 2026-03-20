#include <deal.II/base/function.h>
#include <deal.II/base/types.h>
#include <deal.II/base/mpi_stub.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/sparse_direct.h>

#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/lac/linear_operator.h>
#include <deal.II/lac/packaged_operation.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <unistd.h>
//#include <format>

#include "mesh/mesh.h"
#include "mesh/mesh_processor.h"
#include "timer.h"

using namespace dealii::Functions;

template<int dim>
void write_out_solution(
    dealii::DoFHandler<dim>& dof_handler,
    dealii::BlockVector<double>& solution,
    dealii::BlockVector<double>& prev_solution,
    dealii::Vector<float>& cell_errors,
    unsigned int iter,
    std::string folder
) {
    dealii::DataOut<2> data_out;
    data_out.attach_dof_handler(dof_handler);

    dealii::BlockVector<double> diff(solution);
    diff -= prev_solution;

    std::cout << "max diff block ";
    for (unsigned int b = 0; b < diff.n_blocks(); ++b) {
        std::cout << b << ": " << diff.block(b).linfty_norm() << " ";
    }
    std::cout << "\n";

    data_out.add_data_vector(solution, "solution");
    data_out.add_data_vector(diff, "diff");
    data_out.add_data_vector(cell_errors, "error_per_cell");

    std::ostringstream oss;
    oss << folder << "/solution" << std::setw(2) << std::setfill('0') << iter << ".vtu";
    std::string file = oss.str();
    std::ofstream output(file);

    data_out.build_patches();
    data_out.write_vtu(output);
    output.close();

    std::cout << file << " out\n";
}

struct IdFunction {
    const std::vector<unsigned int> ids;
    const std::vector<double> values;
    const double base_value;

    IdFunction(
        std::vector<unsigned int>&& ids,
        std::vector<double>&& values,
        double base_value = 0
    ) : ids(ids), values(values), base_value(base_value) {
        assert(ids.size() == values.size());
    }

    double operator()(unsigned int id) const {
        for (unsigned int i = 0; i < ids.size(); i++) {
            if (ids[i] == id) {
                return values[i];
            }
        }

        return base_value;
    }
};

void assembly(
    dealii::DoFHandler<2>& dof_handler,
    const dealii::AffineConstraints<double>& constraints,
    dealii::BlockSparseMatrix<double>& system_matrix,
    dealii::BlockVector<double>& system_rhs,
    const dealii::FEValuesExtractors::Scalar& potential,
    const dealii::FEValuesExtractors::Vector& flux,
    const IdFunction& permittivity,
    const IdFunction& boundary_potential
) {
    const unsigned int dim = 2;
    const auto& fe = dof_handler.get_fe();

    const dealii::QGauss<dim> quadrature(fe.degree + 2);
    dealii::FEValues<dim> fe_values(fe, quadrature,
        dealii::update_values | dealii::update_gradients |
        dealii::update_quadrature_points | dealii::update_JxW_values
    );

    const dealii::QGauss<dim - 1> face_quadrature(fe.degree + 2);
    dealii::FEFaceValues<dim> fe_face_values(fe, face_quadrature,
        dealii::update_values | dealii::update_normal_vectors |
        dealii::update_quadrature_points |
        dealii::update_JxW_values
    );

    dealii::FullMatrix<double> local_matrix(fe.dofs_per_cell, fe.dofs_per_cell);
    dealii::Vector<double>     local_rhs(fe.dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(fe.dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);

        local_matrix = 0;
        local_rhs    = 0;

        double eps = permittivity(cell->material_id());

        for (unsigned int q = 0; q < quadrature.size(); q++) {
            for (unsigned int i = 0; i < fe.dofs_per_cell; i++) {

                const auto pot_i = fe_values[potential].value(i, q);
                const auto flux_i = fe_values[flux].value(i, q);
                const auto flux_div_i = fe_values[flux].divergence(i, q);

                for (unsigned int j = 0; j < fe.dofs_per_cell; ++j) {
                    const auto pot_j = fe_values[potential].value(j, q);
                    const auto flux_j = fe_values[flux].value(j, q);
                    const auto flux_div_j = fe_values[flux].divergence(j, q);

                    local_matrix(i, j) += flux_i * flux_j / eps * fe_values.JxW(q);
                    local_matrix(i, j) -= pot_i * flux_div_j * fe_values.JxW(q);
                    local_matrix(i, j) -= flux_div_i * pot_j * fe_values.JxW(q);
                }
            }

            // zero charge
        }

        for (const auto &face : cell->face_iterators()) {
            if (!face->at_boundary()) continue;

            fe_face_values.reinit(cell, face);

            for (unsigned int q = 0; q < face_quadrature.size(); q++) {
                for (unsigned int j = 0; j < fe.dofs_per_cell; j++) {
                    const auto flux_j = fe_face_values[flux].value(j, q);

                    local_rhs(j) -= boundary_potential(face->boundary_id()) 
                        * flux_j * fe_face_values.normal_vector(q)
                        * fe_face_values.JxW(q);

                }
            }
        }

        cell->get_dof_indices(local_dof_indices);
        constraints.distribute_local_to_global(local_matrix, local_rhs, local_dof_indices, system_matrix, system_rhs);
    }
}

void solver(
    dealii::DoFHandler<2>& dof_handler,
    dealii::BlockSparseMatrix<double>& system_matrix,
    dealii::BlockVector<double>& system_rhs,
    dealii::BlockVector<double>& solution
) {
    const auto &M = system_matrix.block(0, 0); // flux mass block
    const auto &B = system_matrix.block(0, 1); // divergence coupling

    const auto &F = system_rhs.block(0); // flux RHS
    const auto &G = system_rhs.block(1); // potential RHS

    auto &U = solution.block(0); // flux unknowns
    auto &P = solution.block(1); // potential unknowns

    const auto op_M = linear_operator(M);
    dealii::ReductionControl reduction_control_M(2000, 1.0e-18, 1.0e-10);
    dealii::SolverCG<dealii::Vector<double>> solver_M(reduction_control_M);
    dealii::PreconditionJacobi<dealii::SparseMatrix<double>> preconditioner_M;
    preconditioner_M.initialize(M);

    const auto op_M_inv = inverse_operator(op_M, solver_M, preconditioner_M);

    const auto op_B = linear_operator(B);
    const auto op_S  = transpose_operator(op_B) * op_M_inv * op_B;

    const auto op_aS = transpose_operator(op_B) * linear_operator(preconditioner_M) * op_B;
    dealii::IterationNumberControl iteration_control_aS(30, 1.e-18);
    dealii::SolverCG<dealii::Vector<double>> solver_aS(iteration_control_aS);
    const auto preconditioner_S = inverse_operator(op_aS, solver_aS, dealii::PreconditionIdentity());

    const auto schur_rhs = transpose_operator(op_B) * op_M_inv * F - G;

    dealii::SolverControl solver_control_S(2000, 1.e-12);
    dealii::SolverCG<dealii::Vector<double>> solver_S(solver_control_S);
    const auto op_S_inv = inverse_operator(op_S, solver_S, preconditioner_S);

    P = op_S_inv * schur_rhs;

    std::cout << solver_control_S.last_step()
              << " CG Schur complement iterations for potential." << std::endl;

    U = op_M_inv * (F - op_B * P);
}

std::vector<dealii::types::global_dof_index> block_sizes(
    const dealii::DoFHandler<2>& dof_handler
) {
    const std::vector<dealii::types::global_dof_index> dofs_per_component = 
        dealii::DoFTools::count_dofs_per_fe_component(dof_handler);
    const unsigned int n_u = dofs_per_component[0], n_p = dofs_per_component[2];
    return {n_u, n_p};
}

void solve_reactor_potential_mixed_method(
    dealii::DoFHandler<2>& dof_handler,
    dealii::BlockVector<double>& solution,
    const dealii::FEValuesExtractors::Scalar& potential,
    const dealii::FEValuesExtractors::Vector& flux
) {
    const unsigned int dim = 2;
    auto& fe = dof_handler.get_fe();

    dealii::AffineConstraints<double> constraints;
    dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);

    dealii::ComponentMask rt_mask = fe.component_mask(flux);
    dealii::DoFTools::make_zero_boundary_constraints(dof_handler, 0, constraints, rt_mask);

    constraints.close();

    dealii::BlockSparsityPattern sparsity_pattern;
    dealii::BlockSparseMatrix<double> system_matrix;
    dealii::BlockVector<double> system_rhs;

    auto permittivity = IdFunction(
        {WATER_MAT_ID, AIR_MAT_ID, WEDGE_MAT_ID},
        {water_permitivity, air_permitivity, wedge_permitivity});

    auto boundary_potential = IdFunction(
        {ELECTRODE1_BOUNDARY_ID, ELECTRODE2_BOUNDARY_ID},
        {V1, V2});

    {
        Timer timer("System initialization: ");

        const auto sizes = block_sizes(dof_handler);
        dealii::BlockDynamicSparsityPattern dsp(sizes, sizes);
        dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp);

        sparsity_pattern.copy_from(dsp);
        system_matrix.reinit(sparsity_pattern);
        system_rhs.reinit(sizes);
    }

    {
        Timer timer("Assembly: ");
        assembly(dof_handler, constraints, system_matrix, system_rhs, potential, flux, permittivity, boundary_potential);
    }

    //{
    //    Timer timer("solver: ");
    //    solver(dof_handler, system_matrix, system_rhs, solution);
    //}

    {
        Timer timer("direct solver: ");
        solution = system_rhs;
        dealii::SparseDirectUMFPACK A_direct;
        A_direct.solve(system_matrix, solution);

    }
}


void compute_reactor_potential_mixed_method(float refine_level) {
    const unsigned int dim = 2;

    std::string folder = "reactor_solutions_mixed";
    if (refine_level != 1) {
        std::ostringstream oss;
        oss << folder << "_" << std::fixed << std::setprecision(1) << refine_level << "adaptive";
        folder = oss.str();
    }

    auto triangulation = build_triangulation();
    triangulation.set_mesh_smoothing( dealii::Triangulation<2>::limit_level_difference_at_vertices );

    const dealii::FESystem<dim> fe (
        dealii::FE_RaviartThomas<dim>(0),
        dealii::FE_DGQ<dim>(0)
    );

    dealii::DoFHandler<dim> dof_handler{triangulation};
    dof_handler.distribute_dofs(fe);
    dealii::DoFRenumbering::component_wise(dof_handler);

    const dealii::FEValuesExtractors::Vector flux(0);
    const dealii::FEValuesExtractors::Scalar potential(dim);

    dealii::BlockVector<double> solution(block_sizes(dof_handler));
    dealii::BlockVector<double> prev_solution(block_sizes(dof_handler));

    for (int i = 0; i < 11; i++) {

        std::cout << "\n";

        solution = prev_solution;
        solve_reactor_potential_mixed_method(dof_handler, solution, potential, flux);

        triangulation.prepare_coarsening_and_refinement();

        dealii::Vector<float> error_per_cell(triangulation.n_active_cells());
        dealii::SolutionTransfer<2, dealii::BlockVector<double>> solution_transfer(dof_handler);
        solution_transfer.prepare_for_coarsening_and_refinement(solution);

        //if (refine_level < 1.) {
        //    Timer timer("Calculating residuals: ");
        //    calculate_poisson_face_residual(dof_handler, solution, error_per_cell,
        //            permittivity, AllNonBoundaryFacesPredicate<2>());
        //    apply_elementwise(error_per_cell, [](float x){ return std::sqrt(x); });
        //}

        {
            Timer timer("Writing to files: ");
            write_out_solution(dof_handler, solution, prev_solution, error_per_cell, i, folder); 
        }

        if (refine_level < 1.) {
            Timer timer("Executing adaptive refinement: ");
            dealii::GridRefinement::refine_and_coarsen_fixed_number(triangulation, error_per_cell, refine_level, 0.0);
            triangulation.execute_coarsening_and_refinement();
        }

        if (refine_level == 1.) {
            Timer timer("Executing global refinement: ");
            triangulation.refine_global(1);
        }

        {
            Timer timer("Transfering solution: ");

            dof_handler.distribute_dofs(fe);
            dealii::DoFRenumbering::component_wise(dof_handler);

            prev_solution.reinit(block_sizes(dof_handler));
            solution_transfer.interpolate(prev_solution);
        }

    }
}

int main(int argc, char **argv) {
    compute_reactor_potential_mixed_method(1);
}

