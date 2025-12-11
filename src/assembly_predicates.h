#pragma once

#include <deal.II/base/types.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/tria_accessor.h>
#include <concepts>

template<int dim, typename T>
concept CellPredicateConcept = requires(T pred, dealii::CellAccessor<dim> cell) {
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
struct AlwaysTrueCellPredicate {
    constexpr bool operator()(const dealii::CellAccessor<dim>& cell) const {
        return true;
    }
};


template<int dim, typename T>
concept FacePredicateConcept = requires(T pred, dealii::CellAccessor<dim> cell, unsigned int face) {
    { pred(cell, face) } -> std::convertible_to<bool>;
};


template<int dim>
struct AlwaysTrueBoundaryPredicate {
    constexpr bool operator()(const dealii::CellAccessor<dim>& cell, unsigned int face) const {
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
        auto neighbor = cell.neighbor(face);
        return neighbor->material_id() == mat2;
    }
};



