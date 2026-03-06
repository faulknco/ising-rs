#[cfg(feature = "cuda")]
pub mod cuda;

pub mod coarsening;
pub mod graph;
pub mod fitting;
pub mod kibble_zurek;
pub mod fss;
pub mod lattice;
pub mod metropolis;
pub mod observables;
pub mod sweep;
pub mod wasm;
pub mod wolff;
