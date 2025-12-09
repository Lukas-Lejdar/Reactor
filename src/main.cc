
#include <deal.II/base/function.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/manifold_lib.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <string>
#include <format>

#include "poisson.h"

struct RadialCapacitor{
    const double r0;
    const double r1;
    const double r2;

    const double voltage0;
    const double surface_charge1;

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
    // phi(r ∈ [r0, r1]) = solution[0] ln r + solution[1]
    // phi(r ∈ [r1, r2]) = solution[2] ln r + solution[3]
    dealii::Vector<double> consts; 

public:
    const RadialCapacitor capacitor;
    Exact2DPotentialSolution(const RadialCapacitor capacitor) 
        : capacitor(capacitor) 
    {
        const double rhs_vec[4] = { capacitor.voltage0, capacitor.surface_charge1, 0, 0 };
        const double system_mat[4][4] = {
            { std::log(capacitor.r0), 1.0, 0.0, 0.0 },
            { 0.0, 0.0, 1/(capacitor.r2), 0.0 },
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

    const dealii::Vector<double>& get_consts() const { return consts; }

};

template<int dim>
void change_boundary_id(
    dealii::Triangulation<dim>& triangulation,
    const dealii::types::boundary_id from,
    const dealii::types::boundary_id to
) {
    for (auto &cell : triangulation.active_cell_iterators()) {
        for (uint f = 0; f < dealii::GeometryInfo<2>::faces_per_cell; ++f) {
            if (cell->face(f)->at_boundary() && cell->face(f)->boundary_id() == from) {
                cell->face(f)->set_boundary_id(to);
            }
        }
    }
}

const dealii::types::boundary_id INNER_ID = 1, OUTER_ID = 2, SIDES_ID = 3; 

template<int dim>
dealii::Triangulation<dim> create_capacitor_triangulation(const RadialCapacitor& capacitor) {
    dealii::Triangulation<dim> triangulation;
    const dealii::Point<dim> center(0, 0);
    dealii::GridGenerator::hyper_shell( triangulation, center, capacitor.r0, capacitor.r2, 10, true);
    change_boundary_id(triangulation, 1, OUTER_ID);
    change_boundary_id(triangulation, 0, INNER_ID);
    return triangulation;
} 

dealii::Triangulation<2> import_capacitor_triangulation(std::string file) {
    dealii::Triangulation<2> triangulation;
    dealii::GridIn<2> gridin;
    gridin.attach_triangulation(triangulation);
    std::ifstream input_file(file);
    gridin.read_msh(input_file);  // Gmsh format

    dealii::Point<2> center(0.0, 0.0);
    triangulation.set_manifold(INNER_ID, dealii::SphericalManifold<2>(center));
    triangulation.set_all_manifold_ids_on_boundary(INNER_ID);
    triangulation.set_manifold(OUTER_ID, dealii::SphericalManifold<2>(center));
    triangulation.set_all_manifold_ids_on_boundary(OUTER_ID);

    return triangulation;
} 

template <int dim>
unsigned int count_faces_with_boundary_id(const dealii::Triangulation<dim> &triangulation, const unsigned int boundary_id) {
    unsigned int count = 0;
    for (const auto &cell : triangulation.active_cell_iterators())
        for (unsigned int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f)
            if (cell->face(f)->at_boundary() &&
                cell->face(f)->boundary_id() == boundary_id)
                ++count;
    return count;
}


int main() {

    // problem definition

    const RadialCapacitor capacitor{0.5, 0.75, 1., 0., 0.75, 1., 2.}; // r0, r1, r2, U0, U2, eps0_1, eps1_2
    const Exact2DPotentialSolution ex_solution(capacitor); 
    const PermittivityFunction<2> permittivity{capacitor};

    // Create triangulation

    //dealii::Triangulation<2> triangulation = create_capacitor_triangulation<2>(capacitor);
    dealii::Triangulation<2> triangulation = import_capacitor_triangulation("../capacitor.msh");

    // solve

    dealii::FE_Q<2> fe{1};
    dealii::DoFHandler<2> dof_handler{triangulation};
    dof_handler.distribute_dofs(fe);

    std::ofstream error_file("l2_errors.txt");

    for (int i = 0; i < 12; i++) {

        dealii::AffineConstraints<double> constraints;
        dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);
        dealii::VectorTools::interpolate_boundary_values(dof_handler, INNER_ID, dealii::Functions::ConstantFunction<2>(capacitor.voltage0), constraints); 
        constraints.close();

        auto poisson_system = LinearSystem(dof_handler, constraints);

        assemble_poisson_system(dof_handler, constraints, permittivity, poisson_system.matrix, poisson_system.rhs);
        assemble_poisson_rhs(dof_handler, permittivity, OUTER_ID, dealii::Functions::ConstantFunction<2>(ex_solution.get_consts()(2)/capacitor.r2), poisson_system.rhs);
        assemble_poisson_rhs(dof_handler, permittivity, SIDES_ID, dealii::Functions::ZeroFunction<2>(), poisson_system.rhs);

        auto solution = solve_cg(poisson_system.matrix, poisson_system.rhs);
        constraints.distribute(solution);

        // out

        write_out_solution(dof_handler, solution, std::format("solutions/solution{:02}.vtu", i));

        // l2 error

        double l2_error = get_l2_error(dof_handler, solution, ex_solution);
        double cell_size = smallest_cell_size(triangulation);
        std::cout << "smallest cell size: " << cell_size << " l2: " << l2_error << "\n";
        error_file << cell_size << " " << l2_error << "\n";
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
