#pragma once

#include <cassert>
#include <deal.II/base/types.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/fe/fe_values.h>
#include <concepts>
#include <vector>

template<int dim, typename T>
concept CellPredicateConcept = requires(T pred, const dealii::CellAccessor<dim> cell) {
    { pred(cell) } -> std::convertible_to<bool>;
};


template<int dim>
struct MaterialIDPredicate {
    dealii::types::material_id material_id;
    constexpr bool operator()(const dealii::CellAccessor<dim>& cell) const {
        return cell.material_id() == material_id;
    }
};


template<int dim>
struct AllCellsPredicate {
    constexpr bool operator()(const dealii::CellAccessor<dim>& cell) const {
        return true;
    }
};


template<int dim, typename T>
concept FacePredicateConcept = requires(T pred, const dealii::CellAccessor<dim> cell, unsigned int face) {
    { pred(cell, face) } -> std::convertible_to<bool>;
};


template<int dim>
struct AllFacesPredicate {
    constexpr bool operator()(const dealii::CellAccessor<dim>& cell, unsigned int face) const {
        return true;
    }
};

template<int dim>
struct AllNonBoundaryFacesPredicate {
    constexpr bool operator()(const dealii::CellAccessor<dim>& cell, unsigned int face) const {
        if (cell.face(face)->at_boundary()) return false;
        return true;
    }
};


template<int dim>
struct BoundaryIDPredicate {
    dealii::types::boundary_id boundary_id;
    constexpr bool operator()(const dealii::CellAccessor<dim>& cell, unsigned int face) const {
        if (!cell.face(face)->at_boundary()) return false;
        return cell.face(face)->boundary_id() == boundary_id;
    }
};


template<int dim>
struct BoundaryAndMaterialPredicate {
    dealii::types::boundary_id boundary_id;
    dealii::types::material_id material_id;
    constexpr bool operator()(const dealii::CellAccessor<dim>& cell, unsigned int face) const {
        if (!cell.face(face)->at_boundary()) return false;
        return cell.material_id() == material_id && cell.face(face)->boundary_id() == boundary_id;
    }
};


template<int dim>
struct InterfacePredicate {
    dealii::types::material_id mat1;
    dealii::types::material_id mat2;
    constexpr bool operator()(const dealii::CellAccessor<dim>& cell, unsigned int face) const {
        if (cell.at_boundary(face)) return false;
        if (cell.material_id() != mat1) return false;
        return cell.neighbor(face)->material_id() == mat2;
    }
};


template<int dim>
struct AllInterfacesPredicate {
    constexpr bool operator()(const dealii::CellAccessor<dim>& cell, unsigned int face) const {
        if (cell.at_boundary(face)) return false;
        if (cell.material_id() != cell.neighbor(face)->material_id()) return true;
        return false;
    }
};

template<int dim>
struct NotBoundaryFaceMaterialPredicate {
    dealii::types::material_id mat1;
    constexpr bool operator()(const dealii::CellAccessor<dim>& cell, unsigned int face) const {
        cell.neighbor_of_neighbor(face);
        if (cell.face(face)->at_boundary()) return false;
        if (cell.material_id() == mat1) return true;
        return false;
    }
};

template<int dim>
struct AllBoundariesPredicate{
    constexpr bool operator()(const dealii::CellAccessor<dim>& cell, unsigned int face) const {
        //return cell.at_boundary(face);
        if (cell.at_boundary(face)) return true;
        //if (cell.material_id() < cell.neighbor(face)->material_id()) return true;
        return false;
    }
};


template<int dim, typename T>
concept VertexPredicateConcept = requires(T pred, const dealii::CellAccessor<dim> cell, unsigned int v) {
    { pred(cell, v) } -> std::convertible_to<bool>;
};


template<int dim>   
struct AllVerticesPredicate {
    constexpr bool operator()(const dealii::CellAccessor<dim>& cell, unsigned int v) {
        return true;
    };
};

template<int dim>   
struct VertexListPredicate {
    const std::vector<int> global_indices;

    VertexListPredicate(const std::vector<int>& global_indices) : global_indices(global_indices) {};

    constexpr bool operator()(const dealii::CellAccessor<dim>& cell, unsigned int v) {
        const dealii::types::global_vertex_index global_v_index = cell.vertex_index(v);
        for (auto v : global_indices) {
            if (v == global_v_index) return true;
        }

        return false;
    };
};


template<int dim, typename T>
concept FEQuadratureFunctionConcept =
requires(T &f, const dealii::CellAccessor<dim>& cell, const dealii::FEValues<dim>& fe_values, unsigned int q, unsigned int i, unsigned int j) 
{
    { f(cell, fe_values, q) } -> std::convertible_to<double>;
};

struct ConstantQuadratureFunction {
    double value;
    ConstantQuadratureFunction(double value) : value(value) {}

    template<int dim, typename FEType>
    double operator()(const dealii::CellAccessor<dim>& cell, const FEType&, unsigned int) const {
        return value;
    }
};

struct MaterialIdQuadratureFunction {
    const std::vector<unsigned int> ids;
    const std::vector<double> values;
    const double base_value;

    MaterialIdQuadratureFunction(
        std::vector<unsigned int>&& ids,
        std::vector<double>&& values,
        double base_value = 0
    ) : ids(ids), values(values), base_value(base_value) {
        assert(ids.size() == values.size());
    }

    template<int dim, typename FEType>
    double operator()(const dealii::CellAccessor<dim>& cell, const FEType&, unsigned int) const {

        for (unsigned int i = 0; i < ids.size(); i++) {
            if (ids[i] == cell.material_id()) {
                return values[i];
            }
        }

        return base_value;
    }

};

