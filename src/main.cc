
#include "assembly/poisson.h"
#include "mesh/mesh_processor.h"

int main(int argc, char **argv) {

    int opt;
    float refine_level = 1;
    int fe_degree = 0;

    while ((opt = getopt(argc, argv, "r:d:")) != -1) {
        switch (opt) {
            case 'r':
                refine_level = std::stof(optarg);
                Assert(refine_level <= 1, dealii::ExcMessage("Refine level must be smaller than 1"));
                Assert(refine_level > 0, dealii::ExcMessage("Refine level can't be negative"));
                break;

            case 'd':
                fe_degree = std::stoi(optarg);
                Assert(fe_degree >= 0, dealii::ExcMessage("Element degree must be non-negative"));
                break;
        }
    }

    auto permittivity = IdFunction(
        {WATER_MAT_ID, AIR_MAT_ID, WEDGE_MAT_ID},
        {water_permitivity, air_permitivity, wedge_permitivity});

    auto boundary_potential = IdFunction(
        {ELECTRODE1_BOUNDARY_ID, ELECTRODE2_BOUNDARY_ID},
        {V1, V2});

    auto triangulation = build_triangulation();
    triangulation.set_mesh_smoothing( dealii::Triangulation<2>::limit_level_difference_at_vertices );

    compute_reactor_potential_mixed_method(triangulation, permittivity, boundary_potential, refine_level, fe_degree);
}

