
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/cell_data.h>
#include <deal.II/grid/manifold_lib.h>

#pragma once

#include "mesh.h"

const double water_permitivity = 78.;
const double air_permitivity = 1.;
const double wedge_permitivity = 2.;

const double V2 = 0.;
const double V1= 10.;

dealii::Triangulation<2> build_triangulation() {
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

    dealii::GridTools::consistently_order_cells(cells);
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


