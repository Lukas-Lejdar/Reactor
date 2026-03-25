
#pragma once

#include <boost/signals2/detail/auto_buffer.hpp>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/types.h>
#include <deal.II/base/mpi_stub.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_postprocessor.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/numerics/data_component_interpretation.h>
#include <deal.II/numerics/data_postprocessor.h>

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
#include <getopt.h>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <unistd.h>
#include <filesystem>

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdint>
#include <string>

#include "timer.h"

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

template<int dim> 
void output_result( 
    dealii::DoFHandler<dim>& dof_handler,
    dealii::BlockVector<double>& solution,
    dealii::BlockVector<double>& prev_solution,
    const IdFunction& permittivity,
    unsigned int iter,
    std::string folder
);


template<int dim>
void output_dof_values(
    dealii::DoFHandler<dim>& dof_handler,
    dealii::BlockVector<double>& solution,
    dealii::BlockVector<double>& prev_solution,
    const IdFunction& permittivity,
    unsigned int iter,
    std::string folder
);

void assembly(
    dealii::DoFHandler<2>& dof_handler,
    const dealii::AffineConstraints<double>& constraints,
    dealii::BlockSparseMatrix<double>& system_matrix,
    dealii::BlockVector<double>& system_rhs,
    const dealii::FEValuesExtractors::Scalar& potential,
    const dealii::FEValuesExtractors::Vector& flux,
    const IdFunction& permittivity,
    const IdFunction& boundary_potential
);

void solver(
    dealii::DoFHandler<2>& dof_handler,
    dealii::BlockSparseMatrix<double>& system_matrix,
    dealii::BlockVector<double>& system_rhs,
    dealii::BlockVector<double>& solution
);

void solve_reactor_potential_mixed_method(
    dealii::DoFHandler<2>& dof_handler,
    dealii::BlockVector<double>& solution,
    const dealii::FEValuesExtractors::Scalar& potential,
    const dealii::FEValuesExtractors::Vector& flux,
    const IdFunction& permittivity,
    const IdFunction& boundary_potential
);

void error_estimator(
    const dealii::DoFHandler<2>& dof_handler,
    const dealii::BlockVector<double>& solution,
    dealii::Vector<float>& errors_per_cell,
    const dealii::FEValuesExtractors::Scalar& potential,
    const dealii::FEValuesExtractors::Vector& flux,
    const IdFunction& permittivity
);

void compute_reactor_potential_mixed_method(
    dealii::Triangulation<2>& triangulation,
    const IdFunction& permittivity,
    const IdFunction& boundary_potential,
    float refine_level = 1,
    unsigned int fe_deg = 0
);

