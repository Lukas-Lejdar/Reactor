
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

#include <vector>
#include <string>
#include <iostream>
#include <limits>
#include <cmath>
#include <fstream>

dealii::UpdateFlags POISSON_VOLUME_FLAGS = 
    dealii::update_gradients |
    dealii::update_JxW_values |
    dealii::update_quadrature_points;


template<int dim>
void assemble_local_poisson_volume (
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
    dealii::update_JxW_values  |
    dealii::update_quadrature_points |
    dealii::update_gradients | 
    dealii::update_normal_vectors;


template<int dim>
void assemble_local_poisson_boundary (
    const dealii::FEFaceValues<dim>& fe_fvalues,
    const dealii::Function<dim>& permittivity,
    dealii::FullMatrix<double> &local_mat
) {
    const auto flags = fe_fvalues.get_update_flags();
    Assert((flags & POISSON_BOUNDARY_FLAGS) == POISSON_BOUNDARY_FLAGS,
       dealii::ExcMessage("assemble_local_poisson_boundary_matrix: FEValues missing required update flags."));

    for (const uint q : fe_fvalues.quadrature_point_indices()) {
        const dealii::Point<dim> &x_q = fe_fvalues.quadrature_point(q);
        const double eps = permittivity.value(x_q);
        const auto& normal = fe_fvalues.normal_vector(q);

        for (uint i = 0; i < fe_fvalues.get_fe().dofs_per_cell; ++i) {
            for (uint j = 0; j < fe_fvalues.get_fe().dofs_per_cell; ++j) {
                local_mat(i, j) -= fe_fvalues.shape_value(i, q) * eps * (fe_fvalues.shape_grad(j, q) * normal) * fe_fvalues.JxW(q);
            }
        }
    }
}


dealii::UpdateFlags POISSON_NEUMANN_FLAGS = 
    dealii::update_values |
    dealii::update_JxW_values  |
    dealii::update_quadrature_points;


template<int dim>
void assemble_local_poisson_neuman_condition(
    const dealii::FEFaceValues<dim>& fe_fvalues,
    const dealii::Function<dim>& permittivity,
    const dealii::Function<dim>& flux,
    dealii::Vector<double>& cell_rhs
) {
    const auto flags = fe_fvalues.get_update_flags();
    Assert((flags & POISSON_NEUMANN_FLAGS) == POISSON_NEUMANN_FLAGS,
       dealii::ExcMessage("assemble_local_poisson_boundary_matrix: FEValues missing required update flags."));

    for (const uint q : fe_fvalues.quadrature_point_indices()) {
        const dealii::Point<dim> &x_q = fe_fvalues.quadrature_point(q);
        const double eps = permittivity.value(x_q);

        for (uint i = 0; i < fe_fvalues.get_fe().dofs_per_cell; ++i) {
            cell_rhs(i) += fe_fvalues.shape_value(i, q) * eps * flux.value(x_q) * fe_fvalues.JxW(q);
        }
    }
}


dealii::UpdateFlags POISSON_BOUNDARY_SOURCE_FLAGS = 
    dealii::update_values |
    dealii::update_JxW_values  |
    dealii::update_quadrature_points;


template<int dim>
void assemble_local_poisson_boundary_source(
    const dealii::FEFaceValues<dim>& fe_face_values,
    const dealii::Function<dim>& surface_source,
    dealii::Vector<double>& cell_rhs
) {
    const auto flags = fe_face_values.get_update_flags();
    Assert((flags & POISSON_BOUNDARY_SOURCE_FLAGS) == POISSON_BOUNDARY_SOURCE_FLAGS,
       dealii::ExcMessage("assemble_local_poisson_surface_source: FEValues missing required update flags."));

    for (const uint q : fe_face_values.quadrature_point_indices()) {
        const dealii::Point<dim> &x_q = fe_face_values.quadrature_point(q);
        const double charge = surface_source.value(x_q);

        for (uint i = 0; i < fe_face_values.get_fe().dofs_per_cell; ++i) {
            cell_rhs(i) += charge * fe_face_values.shape_value(i, q) * fe_face_values.JxW(q);
        }
    }
}

