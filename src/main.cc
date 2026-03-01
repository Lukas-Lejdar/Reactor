
#include <boost/math/special_functions/math_fwd.hpp>
#include <cstdlib>
#include <deal.II/base/function.h>
#include <deal.II/base/mpi_stub.h>
#include <deal.II/base/types.h>
#include <deal.II/grid/cell_data.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/solution_transfer.h>
// #include <deal.II/numerics/error_estimator.h>

#include <deal.II/numerics/vector_tools_interpolate.h>
#include <fstream>
#include <iostream>
#include <cmath>
#include <iterator>
#include <string>
#include <format>
#include <chrono>
#include <iostream>
#include <cstdlib>
#include <unistd.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/cell_data.h>

#include "poisson_local_assembly.h"
#include "assembly_predicates.h"
#include "mesh.h"
#include "poisson.h"

using namespace dealii::Functions;

dealii::Triangulation<2> build_triangulation(
    const std::vector<std::vector<float>> vertices,
    const std::vector<std::vector<int>> faces,
    const std::vector<int> material_ids,
    const std::vector<std::pair<int, int>> manifold_ids,
    const std::vector<std::pair<int, std::vector<float>>> circle_centers,
    const std::vector<std::pair<std::vector<int>, int>> boundary_ids,
    const std::vector<std::pair<std::vector<int>, int>> boundary_manifold_ids
) {
    dealii::Triangulation<2> triangulation;

    std::vector<dealii::Point<2>> vertices_dealii(vertices.size());
    for (unsigned int i = 0; i < vertices.size(); ++i) {
        vertices_dealii[i] = dealii::Point<2>(vertices[i][0], vertices[i][1]);
    }

    std::vector<dealii::CellData<2>> cells(faces.size());
    for (unsigned int i = 0; i < faces.size(); ++i) {
        cells[i].vertices[0] = faces[i][0];
        cells[i].vertices[1] = faces[i][1];
        cells[i].vertices[2] = faces[i][3];
        cells[i].vertices[3] = faces[i][2];
        cells[i].material_id = i;
    }

    triangulation.create_triangulation(vertices_dealii, cells, dealii::SubCellData());

    for (auto center : circle_centers) {
        dealii::Point<2> point(center.second[0], center.second[1]);
        triangulation.set_manifold(center.first, dealii::SphericalManifold<2>(point));
    }

    for (auto &cell : triangulation.active_cell_iterators()) {
        int idx = cell->material_id();
        cell->set_material_id(material_ids[idx]);
        cell->set_user_index(idx);
        
        dealii::Point<2> center;
        for (unsigned int v = 0; v < dealii::GeometryInfo<2>::vertices_per_cell; ++v) {
            center += cell->vertex(v);
        }
        center /= dealii::GeometryInfo<2>::vertices_per_cell;

        if (center[1] >= 2.5) {
            cell->set_material_id(AIR_MAT_ID);
        }

        for (unsigned int i = 0; i < manifold_ids.size(); i++) {
            if (manifold_ids[i].first == idx) {
                //cell->set_manifold_id(manifold_ids[i].second);
            }
        }
    }

    for (auto &cell : triangulation.active_cell_iterators()) {
        for (unsigned int f = 0; f < dealii::GeometryInfo<2>::faces_per_cell; ++f) {
            int v0 = cell->face(f)->vertex_index(0);
            int v1 = cell->face(f)->vertex_index(1);

            
            if (!cell->face(f)->at_boundary() && cell->neighbor(f)->index() < cell->index())
                continue;

            if (v0 > v1)
                std::swap(v0, v1);


            for (auto be : boundary_ids) {
                if (v0 == be.first[0] && v1 == be.first[1]) {
                    cell->face(f)->set_boundary_id(be.second);
                }
            }

            for (auto me : boundary_manifold_ids) {
                if (v0 == me.first[0] && v1 == me.first[1]) {
                    cell->face(f)->set_manifold_id(me.second);
                }
            }
        }
    }

    return triangulation;
}

auto duration_ms(auto start, auto end) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

const double water_permitivity = 78.;
const double air_permitivity = 1.;
const double wedge_permitivity = 2.;

const double V2 = 10.;
const double V1= 0.;


class Timer {
public:
    using clock = std::chrono::steady_clock;

    explicit Timer(std::string name = "")
        : name(std::move(name)),
          start(clock::now()) {}

    ~Timer() { stop(); }

    void stop() {
        auto end = clock::now();
        auto duration = end - start;

        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();

        std::cout << name << ms << " ms\n";
    }


private:
    std::string name;
    clock::time_point start;
};

void solve_reactor_potential(
    dealii::DoFHandler<2>& dof_handler,
    dealii::Vector<double>& solution
) {

    dealii::AffineConstraints<double> constraints;

    {
        Timer timer("Constraints setup: ");
        dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);
        dealii::VectorTools::interpolate_boundary_values(dof_handler, ELECTRODE2_BOUNDARY_ID, ConstantFunction<2>(V2), constraints);
        dealii::VectorTools::interpolate_boundary_values(dof_handler, ELECTRODE1_BOUNDARY_ID, ConstantFunction<2>(V1), constraints);
        constraints.close();
    }

    auto system = LinearSystem(dof_handler, constraints);

    {
        Timer timer("Assembly: ");

        assemble_poisson_volume(dof_handler, constraints, system.matrix, system.rhs,
                ConstantQuadratureFunction(water_permitivity),
                MaterialIDPredicate<2>{.material_id=WATER_MAT_ID});

        assemble_poisson_volume(dof_handler, constraints, system.matrix, system.rhs,
                ConstantQuadratureFunction(air_permitivity),
                MaterialIDPredicate<2>{.material_id=AIR_MAT_ID});

        assemble_poisson_volume(dof_handler, constraints, system.matrix, system.rhs,
                ConstantQuadratureFunction(wedge_permitivity),
                MaterialIDPredicate<2>{.material_id=WEDGE_MAT_ID});
    }

    {
        Timer timer("Solver: ");
        solve_cg(system.matrix, system.rhs, solution);
        constraints.distribute(solution);

    }
}


template <typename VectorType, typename UnaryFunction>
void apply_elementwise(VectorType &in, UnaryFunction f) {
    for (unsigned int i = 0; i < in.size(); ++i) {
        in(i) = f(in[i]);
    }
}


void compute_reactor_potential(float refine_level) {

    std::string folder = "reactor_solutions";
    if (refine_level != 1) {
        folder = folder + std::format("_{:.1f}adaptive", refine_level);
    }

    auto triangulation = build_triangulation(vertices, faces, material_ids, manifold_ids, circle_centers, boundary_ids, boundary_manifold_ids);
    triangulation.set_mesh_smoothing( dealii::Triangulation<2>::limit_level_difference_at_vertices );

    std::cout << "built triangulation\n";

    dealii::FE_Q<2> fe{1};
    dealii::DoFHandler<2> dof_handler{triangulation};
    dof_handler.distribute_dofs(fe);

    dealii::Vector<double> solution(dof_handler.n_dofs());
    dealii::Vector<double> prev_solution(dof_handler.n_dofs());

    auto permittivity = MaterialIdQuadratureFunction(
                {WATER_MAT_ID, AIR_MAT_ID, WEDGE_MAT_ID},
                {water_permitivity, air_permitivity, wedge_permitivity});

    for (int i = 0; i < 9; i++) {

        std::cout << "\n";

        solution = prev_solution;
        solve_reactor_potential(dof_handler, solution);

        dealii::Vector<float> error_per_cell(triangulation.n_active_cells());
        dealii::SolutionTransfer<2> solution_transfer(dof_handler);
        solution_transfer.prepare_for_coarsening_and_refinement(solution);


        if (refine_level < 1.) {
            Timer timer("Calculating residuals: ");
            calculate_poisson_face_residual(dof_handler, solution, error_per_cell,
                    permittivity, AllNonBoundaryFacesPredicate<2>());

            apply_elementwise(error_per_cell, [](float x){ return std::sqrt(x); });
        }

        {
            Timer timer("Writing to files: ");
            write_out_solution( dof_handler, solution, prev_solution, error_per_cell, i, folder); 
        }

        if (refine_level < 1.) {
            Timer timer("Executing adaptive refinement: ");
            dealii::GridRefinement::refine_and_coarsen_fixed_number(triangulation, error_per_cell, refine_level, 0.0);
            triangulation.execute_coarsening_and_refinement();
        }

        if (refine_level == 1.) {
            Timer timer("Executing global refinement: ");

            //MPI_Comm mpi_communicator = MPI_COMM_WORLD;
            //dealii::parallel::distributed::Triangulation<2> parallel_triangulation(mpi_communicator);
            //parallel_triangulation.copy_triangulation(triangulation);
            //triangulation.copy_triangulation(parallel_triangulation);
        
            triangulation.refine_global(1);
        }


        {
            Timer timer("Transfering solution: ");
            dof_handler.distribute_dofs(fe);
            prev_solution.reinit(dof_handler.n_dofs());
            solution_transfer.interpolate(prev_solution);
        }

    }
}

int main(int argc, char **argv) {

    int opt;
    float refine_level = 1;

    while ((opt = getopt(argc, argv, "r:")) != -1) {
        switch (opt) {
            case 'r':
                refine_level = std::stof(optarg);
                Assert(refine_level <= 1, "Refine level must be smaller than 1");
                Assert(refine_level < 0, "Refine level can't be negative");
                break;
        }
    }

    //auto triangulation = build_triangulation(vertices, faces, material_ids, manifold_ids, circle_centers, boundary_ids, boundary_manifold_ids);

    //std::cout << "start\n";

    //auto triangulation = build_triangulation();

    //std::cout << "triangulation built \n";
    //triangulation.refine_global(2);

    //dealii::GridOut grid_out;
    //std::ofstream out("exported.msh");
    //grid_out.write_msh(triangulation, out);

    compute_reactor_potential(refine_level);

    //improve_mesh_winslow();
}

