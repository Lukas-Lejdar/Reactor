
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
#include <deal.II/numerics/vector_tools_constraints.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
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

const dealii::types::boundary_id inner_id = 0; 
const dealii::types::boundary_id outer_id = 1; 

const float voltage0 = 0.;
const float voltage2 = 1.;

template<int dim>
class Poisson {
    public:
        Poisson(dealii::Triangulation<dim>&);

        void reinitialize();
        void add_volume_matrix(dealii::SparseMatrix<double>& system_matrix, dealii::Vector<double>& rhs, const dealii::Function<dim>& permittivity);
        void add_boundary_matrix(dealii::SparseMatrix<double>& system_matrix, dealii::Vector<double>& rhs, const dealii::Function<dim>& permittivity);
        void write_out_solution(const dealii::Vector<double>& solution, std::string file);

        const dealii::FE_Q<dim> fe;
        const dealii::Triangulation<dim>& triangulation;

        const dealii::SparsityPattern& get_sp() const { return sp; }
        const dealii::DoFHandler<dim>& get_dof_handler() const { return dof_handler; }

        dealii::AffineConstraints<double> constraints;

    private:

        dealii::DoFHandler<dim> dof_handler;
        dealii::SparsityPattern sp;

        dealii::QGauss<dim> quadrature;
        dealii::QGauss<dim-1> face_quadrature;
};

template <int dim>
Poisson<dim>::Poisson(dealii::Triangulation<dim>& triangulation)
  : fe(1)
  , triangulation(triangulation)
  , dof_handler(triangulation)
  , quadrature(fe.degree + 1)
  , face_quadrature(fe.degree + 1)
{
    reinitialize();
}

template <int dim>
void Poisson<dim>::reinitialize() {
    dof_handler.distribute_dofs(fe);
    constraints.clear();
    
    dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    dealii::VectorTools::interpolate_boundary_values(dof_handler, inner_id, dealii::Functions::ConstantFunction<dim>(voltage0), constraints); 
    dealii::VectorTools::interpolate_boundary_values(dof_handler, outer_id, dealii::Functions::ConstantFunction<dim>(voltage2), constraints); 

    constraints.close();

    dealii::DynamicSparsityPattern dsp(dof_handler.n_dofs());
    dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);
    sp.copy_from(dsp);
}

template<int dim>
void Poisson<dim>::add_volume_matrix(
    dealii::SparseMatrix<double>& system_matrix,
    dealii::Vector<double>& system_rhs,
    const dealii::Function<dim>& permittivity
) {

    dealii::FEValues<dim> fe_values{fe, quadrature,
        dealii::update_values |
        dealii::update_gradients |
        dealii::update_JxW_values |
        dealii::update_quadrature_points
    };

    dealii::FullMatrix<double> local_mat(fe.dofs_per_cell, fe.dofs_per_cell);
    dealii::Vector<double> cell_rhs(fe.dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof(fe.dofs_per_cell);

    for (const auto& cell : dof_handler.active_cell_iterators()) {
        fe_values.reinit(cell);

        local_mat = 0;
        cell_rhs = 0;

        for (const uint q : fe_values.quadrature_point_indices()) {
            const dealii::Point<dim> &x_q = fe_values.quadrature_point(q);
            for (uint i = 0; i < fe.dofs_per_cell; i++) {
                for (uint j = 0; j < fe.dofs_per_cell; j++) {
                    local_mat(i, j) += fe_values.shape_grad(i, q) 
                        * permittivity.value(x_q) 
                        * fe_values.shape_grad(j, q) 
                        * fe_values.JxW(q);
                }
            }
        }

        cell->get_dof_indices(local_dof);
        constraints.distribute_local_to_global(local_mat, cell_rhs, local_dof, system_matrix, system_rhs);
        //constraints.distribute_local_to_global(local_mat, local_dof, system_matrix);
    }


};

template<int dim>
void Poisson<dim>::add_boundary_matrix(
    dealii::SparseMatrix<double>& system_matrix,
    dealii::Vector<double>& system_rhs,
    const dealii::Function<dim>& permittivity
) {
    dealii::FEFaceValues<dim> fe_face_values{fe, face_quadrature,
        dealii::update_values |
        dealii::update_gradients |
        dealii::update_normal_vectors |
        dealii::update_JxW_values  |
        dealii::update_quadrature_points 
    };

    dealii::FullMatrix<double> local_mat(fe.dofs_per_cell, fe.dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof(fe.dofs_per_cell);
    dealii::Vector<double> cell_rhs(fe.dofs_per_cell);

    for (const auto& cell : dof_handler.active_cell_iterators()) {
        local_mat = 0;
        cell_rhs = 0;

        for (uint face = 0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
            if (!cell->face(face)->at_boundary()) continue;
            fe_face_values.reinit(cell, face);

            for (uint q = 0; q < face_quadrature.size(); ++q) {
                const dealii::Point<dim> &x_q = fe_face_values.quadrature_point(q);
                for (uint i = 0; i < fe.dofs_per_cell; ++i) {
                    for (uint j = 0; j < fe.dofs_per_cell; ++j) {
                        const dealii::Tensor<1,dim>& normal = fe_face_values.normal_vector(q);
                        local_mat(i, j) -= 
                            fe_face_values.shape_value(i, q) 
                            * normal 
                            * permittivity.value(x_q) 
                            * fe_face_values.shape_grad(j, q) 
                            * fe_face_values.JxW(q);
                    }
                }
            }
        }

        cell->get_dof_indices(local_dof);
        constraints.distribute_local_to_global(local_mat, cell_rhs, local_dof, system_matrix, system_rhs);
    }
}

template<int dim>
void Poisson<dim>::write_out_solution(const dealii::Vector<double>& solution, std::string file) {
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


/// Relative permittivity as a function of position
template <int dim>
class PermittivityFunction : public dealii::Function<dim> {
public:
    const RadialCapacitor capacitor;
    PermittivityFunction(const RadialCapacitor capacitor) : capacitor(capacitor) {}

    double value(const dealii::Point<dim> &p, const unsigned int = 0) const override {
        return p.norm() < capacitor.r1 ? capacitor.epsilon0_1 : capacitor.epsilon1_2;
    }
};

template <int dim>
dealii::Vector<double> run(Poisson<dim>& poisson, const RadialCapacitor& capacitor) {

    const auto& dof_handler = poisson.get_dof_handler();

    // setup

    dealii::Vector<double> solution;
    dealii::Vector<double> system_rhs;
    dealii::SparseMatrix<double> system_matrix;

    system_matrix.reinit(poisson.get_sp());
    system_rhs.reinit(dof_handler.n_dofs());
    solution.reinit(dof_handler.n_dofs());

    const PermittivityFunction<2> permittivity{capacitor};

    poisson.add_volume_matrix(system_matrix, system_rhs, permittivity);
    poisson.add_boundary_matrix(system_matrix, system_rhs, permittivity);
    
    // solve

    dealii::SolverControl solver_control(1000, 1e-6 * system_rhs.l2_norm());
    dealii::PreconditionSSOR<dealii::SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);

    dealii::SolverCG<dealii::Vector<double>> solver(solver_control);
    solver.solve(system_matrix, solution, system_rhs, dealii::PreconditionIdentity());

    poisson.constraints.distribute(solution);

    // return 

    std::cout << solver_control.last_step() << " CG iterations needed to obtain convergence." << std::endl;

    return solution;
}

int main() {

    // problem definition

    const RadialCapacitor capacitor{0.5, 0.75, 1., voltage0, voltage2, 1., 10.}; // r0, r1, r2, U0, U2, eps0_1, eps1_2
    const Exact2DPotentialSolution ex_solution(capacitor); 

    // Create triangulation

    dealii::Triangulation<2> triangulation;
    const dealii::Point<2> center(0, 0);
    dealii::GridGenerator::hyper_shell( triangulation, center, capacitor.r0, capacitor.r2, 10);
    triangulation.refine_global(2);

    // label boundaries

    for (const auto &cell : triangulation.active_cell_iterators()) {
        for (unsigned int f = 0; f < dealii::GeometryInfo<2>::faces_per_cell; ++f) {
            if (!cell->face(f)->at_boundary()) continue;
            const double distance = cell->face(f)->center().distance(center);
            if (std::abs(distance - capacitor.r0) < 1e-1) cell->face(f)->set_boundary_id(inner_id);
            else if (std::abs(distance - capacitor.r2) < 1e-1) cell->face(f)->set_boundary_id(outer_id);
        }
    }

    // solve

    Poisson<2> poisson{triangulation};
    dealii::Vector<double> solution;


    for (int i = 0; i < 10; i++) {

        solution = run(poisson, capacitor);
        poisson.write_out_solution(solution, "solution" + std::to_string(i) + ".vtu");

        dealii::Vector<double> difference_per_cell(triangulation.n_active_cells());
        dealii::VectorTools::integrate_difference(
            poisson.get_dof_handler(),
            solution,
            ex_solution,
            difference_per_cell,
            dealii::QGauss<2>(poisson.fe.degree + 1),
            dealii::VectorTools::L2_norm
        );

        dealii::Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
        dealii::KellyErrorEstimator<2>::estimate(
            poisson.get_dof_handler(),
            dealii::QGauss<1>(poisson.fe.degree + 1),
            {},
            solution,
            estimated_error_per_cell
        );

        dealii::GridRefinement::refine_and_coarsen_fixed_number(triangulation, estimated_error_per_cell, 0.3, 0.03);
        triangulation.execute_coarsening_and_refinement();


        double l2_error = difference_per_cell.l2_norm();
        std::cout << i << "th refinement total l2 error: " << l2_error << "\n";

        //dealii::Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
        //for (unsigned int c = 0; c < triangulation.n_active_cells(); ++c)
        //    estimated_error_per_cell[c] = difference_per_cell[c];

        //// adaptive
        //dealii::GridRefinement::refine_and_coarsen_fixed_number( triangulation, estimated_error_per_cell, 0.3, 0.0);
        //triangulation.execute_coarsening_and_refinement();
        
        //triangulation.refine_global(1);

        poisson.reinitialize();

    }

    
}
