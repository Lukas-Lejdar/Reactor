
#pragma once

#include <boost/tuple/detail/tuple_basic.hpp>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/fe/fe_q.h>
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
#include <deal.II/fe/fe_system.h>

#include <filesystem>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <limits>
#include <cmath>
#include <fstream>

#include "assembly_predicates.h"
#include "poisson_local_assembly.h"

template<int dim, typename CellPredicate>
void assemble_poisson(
    const dealii::DoFHandler<dim>& dof_handler,
    const dealii::AffineConstraints<double>& constraints,
    dealii::SparseMatrix<double>& system_matrix,
    dealii::Vector<double>& system_rhs,
    const dealii::Function<dim>& permittivity = dealii::Functions::ConstantFunction<dim>(1.0)
) {}

template<int dim, typename CellPredicate = AllCellsPredicate<dim>, typename Permittivity>
requires CellPredicateConcept<dim, CellPredicate> && FEQuadratureFunctionConcept<dim, Permittivity>
void assemble_poisson_volume(
    const dealii::DoFHandler<dim>& dof_handler,
    const dealii::AffineConstraints<double>& constraints,
    dealii::SparseMatrix<double>& system_matrix,
    dealii::Vector<double>& system_rhs,
    Permittivity&& permittivity = ConstantQuadratureFunction(1.),
    CellPredicate&& cell_predicate = AllCellsPredicate<dim>(),
    const std::vector<unsigned int>& components = {0}
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
        assemble_local_poisson_volume(*cell, fe_values, local_mat, permittivity, components);

        cell->get_dof_indices(local_dof);
        constraints.distribute_local_to_global(local_mat, cell_rhs, local_dof, system_matrix, system_rhs);
    }
}

template<int dim, typename CellPredicate = AllCellsPredicate<dim>, typename Permittivity>
requires CellPredicateConcept<dim, CellPredicate> && FEQuadratureFunctionConcept<dim, Permittivity>
void calculate_poisson_residual(
    const dealii::DoFHandler<dim>& dof_handler,
    dealii::Vector<double>& solution,
    dealii::Vector<double>& errors,
    Permittivity&& permittivity = ConstantQuadratureFunction(1.),
    CellPredicate&& cell_predicate = AllCellsPredicate<dim>()
) {

    auto& fe = dof_handler.get_fe();

    dealii::QGauss<dim> quadrature{fe.degree + 1};
    dealii::FEValues<dim> fe_values{fe, quadrature, dealii::update_hessians | dealii::update_JxW_values | dealii::update_gradients};
    std::vector<dealii::types::global_dof_index> local_dof(fe.dofs_per_cell);

    for (const auto& cell : dof_handler.active_cell_iterators()) {
        if (!cell_predicate(*cell)) continue;
        cell->get_dof_indices(local_dof);

        std::vector<dealii::Tensor<2, dim>> hessians(fe_values.n_quadrature_points);

        fe_values.reinit(cell);
        fe_values.get_function_hessians(solution, hessians);

        for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q) {
            double eps = permittivity(*cell, fe_values, q);
            double residual = eps * dealii::trace(hessians[q]); 
            errors[cell->active_cell_index()] += std::pow(cell->diameter() * residual, 2) * fe_values.JxW(q);
        }
    }
}

template<int dim, typename FacePredicate, typename Permittivity>
requires FacePredicateConcept<dim, FacePredicate> && FEQuadratureFunctionConcept<dim, Permittivity>
void calculate_poisson_face_residual(
    const dealii::DoFHandler<dim>& dof_handler,
    dealii::Vector<double>& solution,
    dealii::Vector<float>& errors,
    Permittivity&& permittivity,
    FacePredicate&& face_predicate
) {

    auto& fe = dof_handler.get_fe();

    dealii::QGauss<dim-1> face_quadrature{fe.degree + 1};

    dealii::FEFaceValues<dim> fe_fvalues( fe, face_quadrature,
        dealii::update_gradients |
        dealii::update_JxW_values |
        dealii::update_normal_vectors |
        dealii::update_quadrature_points );

    dealii::FEFaceValues<dim> fe_fvalues_neighbor( fe, face_quadrature,
            dealii::update_gradients |
            dealii::update_normal_vectors |
        dealii::update_quadrature_points );

    dealii::FESubfaceValues<dim> fe_subface_values( fe, face_quadrature,
        dealii::update_gradients |
        dealii::update_JxW_values |
        dealii::update_normal_vectors |
        dealii::update_quadrature_points );

    for (const auto& cell : dof_handler.active_cell_iterators()) {
        for (uint f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f) {
            if (!face_predicate(*cell, f)) continue;

            double diameter = cell->face(f)->diameter();

            // same refinement level
            if (!cell->face(f)->at_boundary() && 
                !cell->neighbor_is_coarser(f) && !cell->neighbor(f)->has_children()) {

                auto neighbor = cell->neighbor(f);

                fe_fvalues.reinit(cell, f);
                fe_fvalues_neighbor.reinit(neighbor, cell->neighbor_of_neighbor(f));
                
                std::vector<dealii::Tensor<1,dim>> grad_cell(face_quadrature.size());
                std::vector<dealii::Tensor<1,dim>> grad_neighbor(face_quadrature.size());

                fe_fvalues.get_function_gradients(solution, grad_cell);
                fe_fvalues_neighbor.get_function_gradients(solution, grad_neighbor);
                
                for (unsigned int q=0; q<face_quadrature.size(); ++q) {
                    double eps1 = permittivity(*cell, fe_fvalues, q);
                    double eps2 = permittivity(*neighbor, fe_fvalues_neighbor, q);
                    double jump = (eps1 * grad_cell[q] - eps2 * grad_neighbor[q]) * fe_fvalues.normal_vector(q);
                    errors[cell->active_cell_index()] += diameter * jump * jump * fe_fvalues.JxW(q);
                }
            }


            // neighbor is more refined
            if (!cell->face(f)->at_boundary() && cell->neighbor(f)->has_children()) {
                for (unsigned int subface = 0; subface < cell->face(f)->n_children(); subface++) {
                    auto neighbor_child = cell->neighbor_child_on_subface(f, subface);

                    fe_subface_values.reinit(cell, f, subface);
                    fe_fvalues_neighbor.reinit(neighbor_child, cell->neighbor_of_neighbor(f)); // TODO: really cell->neighbor_of_neighbor(f) ?

                    std::vector<dealii::Tensor<1,dim>> grad_cell(face_quadrature.size());
                    std::vector<dealii::Tensor<1,dim>> grad_neighbor(face_quadrature.size());

                    fe_subface_values.get_function_gradients(solution, grad_cell);
                    fe_fvalues_neighbor.get_function_gradients(solution, grad_neighbor);


                    for (unsigned int q=0; q<face_quadrature.size(); ++q) {
                        double eps1 = permittivity(*cell, fe_subface_values, q);
                        double eps2 = permittivity(*neighbor_child, fe_fvalues_neighbor, q);
                        double jump = ( eps1 * grad_cell[q] - eps2 * grad_neighbor[q] ) * fe_fvalues.normal_vector(q);
                        errors[cell->active_cell_index()] += diameter * jump * jump * fe_subface_values.JxW(q);
                    }
                }
            }
        
            // skip coarser neighbors
        }
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
            assemble_local_poisson_neuman_condition(fe_face_values, flux, cell_rhs);
        }

        cell->get_dof_indices(local_dof);
        system_rhs.add(local_dof, cell_rhs);
    }
}


template <int dim, typename FacePredicate>
requires FacePredicateConcept<dim, FacePredicate>
void add_face_dirichlet_conditions(
    const dealii::DoFHandler<dim> &dof_handler,
    dealii::AffineConstraints<double> &constraints,
    const dealii::Function<dim> &boundary_function,
    FacePredicate&& face_predicate,
    const std::vector<unsigned int>& components = {0}
) {

    auto& fe = dof_handler.get_fe();

    for (const auto& cell : dof_handler.active_cell_iterators()) {
        for (uint f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; f++) {
            if(!face_predicate(*cell, f)) continue;
            const auto face = cell->face(f);

            for (unsigned int v = 0; v < dealii::GeometryInfo<dim>::vertices_per_face; v++) {
                const dealii::Point<dim> &vertex = face->vertex(v);

                for (unsigned int d : components) {
                    unsigned int dof_index =  face->vertex_dof_index(v, d);
                    double value = boundary_function.value(vertex, d);

                    constraints.add_line(dof_index);
                    constraints.set_inhomogeneity(dof_index, value);

                }
            }
        }
    }
}

template <typename MatrixType, typename VectorType>
void solve_cg(
    const MatrixType& system_matrix,
    const VectorType& rhs,
    dealii::Vector<double>& solution
) {
    dealii::SolverControl solver_control(4000, 1e-6 * rhs.l2_norm());
    dealii::PreconditionSSOR<MatrixType> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);

    dealii::SolverCG<VectorType> solver(solver_control);
    solver.solve(system_matrix, solution, rhs, preconditioner);
    std::cout << solver_control.last_step() << " CG iterations needed to converge.\n";
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
    for (const auto &cell : triangulation.active_cell_iterators()) {
        h_min = std::min(h_min, cell->minimum_vertex_distance());
    }
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

template <int dim>
class ElectricFieldPostprocessor : public dealii::DataPostprocessorVector<dim> {
    public:
        ElectricFieldPostprocessor(const std::string &name)
            : dealii::DataPostprocessorVector<dim>(name, dealii::update_gradients) {}

        virtual void evaluate_scalar_field (
            const dealii::DataPostprocessorInputs::Scalar<dim> &inputs,
            std::vector<dealii::Vector<double>> &computed_quantities
        ) const override {
            for (unsigned int q = 0; q < inputs.solution_gradients.size(); ++q) {
                for (unsigned int d = 0; d < dim; ++d) {
                    computed_quantities[q][d] = -inputs.solution_gradients[q][d];
                }
            }
        }
};


template<int dim>
void write_out_solution(
    dealii::DoFHandler<dim>& dof_handler,
    dealii::Vector<double>& solution,
    dealii::Vector<double>& prev_solution,
    dealii::Vector<float>& cell_errors,
    unsigned int iter,
    std::string folder
) {
    dealii::DataOut<2> data_out;
    data_out.attach_dof_handler(dof_handler);
    std::vector<std::string> solution_names;

    dealii::Vector<double> diff(solution);
    diff -= prev_solution;

    std::cout << "max potential diff " << diff.linfty_norm() << "\n";

    ElectricFieldPostprocessor<dim> efield("E");
    data_out.add_data_vector(solution, "potential_(V)");
    data_out.add_data_vector(solution, efield);

    ElectricFieldPostprocessor<dim> efield_diff("E_diff");
    data_out.add_data_vector(diff, "potential_diff_(V)");
    data_out.add_data_vector(diff, efield_diff);

    data_out.add_data_vector(cell_errors, "error_per_cell");

    std::string file = folder + std::format("/solution{:02}.vtu", iter);
    std::ofstream output(file);

    data_out.build_patches();
    data_out.write_vtu(output);
    output.close();

    std::cout << file << " out\n";
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

template<int dim>
void refine_hanging_nodes_on_material_interfaces( dealii::Triangulation<dim> &triangulation ) {
    bool changed = true;

    while (changed) {
        changed = false;

        for (auto cell : triangulation.active_cell_iterators()) {
            for (unsigned int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f) {
                if (cell->face(f)->at_boundary()) continue;
                if (cell->material_id() == cell->neighbor(f)->material_id()) continue;

                auto neighbor = cell->neighbor(f);
                if (neighbor->has_children() || neighbor->refine_flag_set() != dealii::RefinementCase<dim>::no_refinement) {
                    if (cell->refine_flag_set() == dealii::RefinementCase<dim>::no_refinement) {
                        cell->set_refine_flag(dealii::RefinementCase<dim>::isotropic_refinement);
                        changed = true;
                    }
                }
            }
        }
    }
}
