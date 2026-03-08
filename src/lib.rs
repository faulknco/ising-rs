#[cfg(feature = "cuda")]
pub mod cuda;

pub mod cli;
pub mod coarsening;
pub mod fitting;
pub mod fss;
pub mod graph;
pub mod heisenberg;
pub mod kibble_zurek;
pub mod lattice;
pub mod metropolis;
pub mod observables;
pub mod sweep;
pub mod wasm;
pub mod wolff;
