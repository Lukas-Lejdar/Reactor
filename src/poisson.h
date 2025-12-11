
#pragma once

#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_integrate_difference.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/numerics/error_estimator.h>

#include <vector>
#include <string>
#include <iostream>
#include <limits>
#include <cmath>
#include <fstream>

#include "assembly_predicates.h"
#include "poisson_local_assembly.h"


template<int dim, typename CellPredicate>
requires CellPredicateConcept<dim, CellPredicate>
void assemble_poisson_volume(
    const dealii::DoFHandler<dim>& dof_handler,
    const dealii::AffineConstraints<double>& constraints,
    dealii::SparseMatrix<double>& system_matrix,
    dealii::Vector<double>& system_rhs,
    const dealii::Function<dim>& permittivity,
    CellPredicate&& cell_predicate
) {
    auto& fe = dof_handler.get_fe();
    
    dealii::QGauss<dim> quadrature{fe.degree + 1};
    dealii::FEValues<dim> fe_values{fe, quadrature, POISSON_VOLUME_FLAGS };

    dealii::FullMatrix<double> local_mat(fe.dofs_per_cell, fe.dofs_per_cell);
    dealii::Vector<double> cell_rhs(fe.dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof(fe.dofs_per_cell);

    for (const auto& cell : dof_handler.active_cell_iterators()) {
        if (!cell_predicate(*cell)) continue;

        local_mat = 0;
        cell_rhs = 0;

        fe_values.reinit(cell);
        assemble_local_poisson_volume(fe_values, permittivity, local_mat);

        cell->get_dof_indices(local_dof);
        constraints.distribute_local_to_global(local_mat, cell_rhs, local_dof, system_matrix, system_rhs);
    }
}


template<int dim>
void assemble_poisson_volume(
    const dealii::DoFHandler<dim>& dof_handler,
    const dealii::AffineConstraints<double>& constraints,
    dealii::SparseMatrix<double>& system_matrix,
    dealii::Vector<double>& system_rhs,
    const dealii::Function<dim>& permittivity
) {
    assemble_poisson_volume(dof_handler, constraints, permittivity, system_matrix, system_rhs,
            AlwaysTrueCellPredicate<dim>{});
}


template<int dim, typename FacePredicate>
requires FacePredicateConcept<dim, FacePredicate>
void assemble_poisson_boundary(
    const dealii::DoFHandler<dim>& dof_handler,
    const dealii::AffineConstraints<double>& constraints,
    dealii::SparseMatrix<double>& system_matrix,
    dealii::Vector<double>& system_rhs,
    const dealii::Function<dim>& permittivity,
    FacePredicate&& face_predicate
) {
    auto& fe = dof_handler.get_fe();
    
    dealii::QGauss<dim-1> face_quadrature{fe.degree + 1};
    dealii::FEFaceValues<dim> fe_face_values{fe, face_quadrature, POISSON_BOUNDARY_FLAGS};

    dealii::FullMatrix<double> local_mat(fe.dofs_per_cell, fe.dofs_per_cell);
    dealii::Vector<double> cell_rhs(fe.dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof(fe.dofs_per_cell);

    for (const auto& cell : dof_handler.active_cell_iterators()) {
        local_mat = 0;
        cell_rhs = 0;

        for (uint face = 0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
            if (!face_predicate(*cell, face)) continue;
            fe_face_values.reinit(cell, face);
            assemble_local_poisson_boundary(fe_face_values, permittivity, local_mat);
        }

        cell->get_dof_indices(local_dof);
        constraints.distribute_local_to_global(local_mat, cell_rhs, local_dof, system_matrix, system_rhs);
    }
}

template<int dim, typename FacePredicate>
requires FacePredicateConcept<dim, FacePredicate>
void assemble_poisson_boundary_source(
    const dealii::DoFHandler<dim>& dof_handler,
    dealii::Vector<double>& system_rhs,
    const dealii::Function<dim>& surface_charge,
    FacePredicate&& face_predicate
) {
    auto& fe = dof_handler.get_fe();
    
    dealii::QGauss<dim-1> face_quadrature{fe.degree + 1};
    dealii::FEFaceValues<dim> fe_face_values{fe, face_quadrature, POISSON_BOUNDARY_SOURCE_FLAGS};

    dealii::Vector<double> cell_rhs(fe.dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof(fe.dofs_per_cell);

    for (const auto& cell : dof_handler.active_cell_iterators()) {
        cell_rhs = 0;

        for (uint face = 0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
            if (!face_predicate(*cell, face)) continue;

            fe_face_values.reinit(cell, face);
            assemble_local_poisson_boundary_source(fe_face_values, surface_charge, cell_rhs);
        }

        cell->get_dof_indices(local_dof);
        system_rhs.add(local_dof, cell_rhs);
    }
}

template<int dim, typename FacePredicate>
requires FacePredicateConcept<dim, FacePredicate>
void assemble_poisson_neuman_condition(
    const dealii::DoFHandler<dim>& dof_handler,
    dealii::Vector<double>& system_rhs,
    const dealii::Function<dim>& permittivity,
    const dealii::Function<dim>& flux,
    FacePredicate&& face_predicate
) {
    auto& fe = dof_handler.get_fe();
    
    dealii::QGauss<dim-1> face_quadrature{fe.degree + 1};
    dealii::FEFaceValues<dim> fe_face_values{fe, face_quadrature, POISSON_NEUMANN_FLAGS};

    dealii::Vector<double> cell_rhs(fe.dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof(fe.dofs_per_cell);

    for (const auto& cell : dof_handler.active_cell_iterators()) {
        cell_rhs = 0;

        for (uint face = 0; face < dealii::GeometryInfo<dim>::faces_per_cell; ++face) {
            if(!face_predicate(*cell, face)) continue;

            fe_face_values.reinit(cell, face);
            assemble_local_poisson_neuman_condition(fe_face_values, permittivity, flux, cell_rhs);
        }

        cell->get_dof_indices(local_dof);
        system_rhs.add(local_dof, cell_rhs);
    }
}


template <typename MatrixType, typename VectorType>
dealii::Vector<double> solve_cg(const MatrixType& system_matrix, const VectorType& rhs) {
    dealii::Vector<double> solution;
    solution.reinit(system_matrix.m());

    dealii::SolverControl solver_control(4000, 1e-6 * rhs.l2_norm());
    dealii::PreconditionSSOR<MatrixType> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);

    dealii::SolverCG<VectorType> solver(solver_control);
    solver.solve(system_matrix, solution, rhs, dealii::PreconditionIdentity());
    std::cout << solver_control.last_step() << " CG iterations needed to converge.\n";

    return solution;
}


class LinearSystem {
public:
    dealii::Vector<double> rhs;
    dealii::SparseMatrix<double> matrix;

    LinearSystem();

    template<int dim>
    LinearSystem( const dealii::DoFHandler<dim>& dof_handler, const dealii::AffineConstraints<double>& constraints) { 
        reinit(dof_handler, constraints); 
    }

    template<int dim>
    void reinit(const dealii::DoFHandler<dim>& dof_handler, const dealii::AffineConstraints<double>& constraints) {
        dealii::DynamicSparsityPattern dsp(dof_handler.n_dofs());
        dealii::DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);

        sp.copy_from(dsp);
        matrix.reinit(sp);
        rhs.reinit(matrix.m());
    }

private:
    dealii::SparsityPattern sp;
};

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

void restrict_refinement_by_cell_size(
    dealii::Triangulation<2> &triangulation,
    const double min_cell_size,
    const double max_cell_size
) {
    for (auto &cell : triangulation.active_cell_iterators()) {
        const double h = cell->diameter();
        if (h < min_cell_size) cell->clear_refine_flag();
        if (h > max_cell_size) cell->clear_coarsen_flag();
    }
}
