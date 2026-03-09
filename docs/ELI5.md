# Explain Like I'm Five: What Does This Project Do?

## The Big Picture

Every magnet you have ever stuck to a fridge is performing a small miracle. Billions of atoms inside it have somehow agreed to point the same direction, producing a magnetic field you can feel with your fingers. Heat that magnet up enough, though, and the atoms lose their coordination. The magnet stops being a magnet.

This project simulates that process on a computer -- atom by atom, flip by flip -- to understand *exactly* how and when magnets fall apart. We wrote the simulation in Rust (a fast, modern programming language) and accelerated the heavy lifting on an NVIDIA GPU using CUDA. Then we compared our numbers to textbook physics and real metals like iron and nickel.

## Imagine a Billion Tiny Magnets

Picture a huge three-dimensional grid of compass needles. Each needle can only point up or down (no in-between -- physicists call this the Ising model). Every needle wants to match its neighbors: if the six needles surrounding it all point up, it feels a strong pull to point up too.

That is the entire model. One simple rule -- "try to match your neighbors" -- and yet it produces shockingly rich behavior when you zoom out to millions of needles at once.

## Hot vs Cold: The Phase Transition

Temperature, in this simulation, is just a measure of how much random jiggling each needle feels. At low temperature the jiggling is weak, neighbors stay aligned, and the whole grid acts like a magnet. At high temperature the jiggling overwhelms the neighbor pull, needles point every which way, and magnetism vanishes.

Think of a frozen lake. Below freezing, water molecules lock into an orderly crystal. Above freezing, they break free into chaotic liquid. The transition from ice to water is sudden and dramatic. Magnets do something similar, and that crossover point is what we are hunting.

## The Magic Temperature

The critical temperature (Tc) is the exact boundary between "magnet" and "not a magnet." Our simulation nails it to four decimal places: **Tc = 4.5115 J/kB**, matching the best theoretical value with an error of about one part in ten thousand. That is like measuring the distance from New York to Los Angeles and being off by half a mile.

Near this temperature, strange things happen. Fluctuations appear at every scale -- tiny clusters and huge clusters of aligned needles coexist, like a lake that is simultaneously freezing and melting everywhere at once. The numbers that describe this behavior are called *critical exponents*, and we measure those too.

## Bigger is Better: Finite-Size Scaling

A real magnet has something like 10^23 atoms. We cannot simulate that many. So we run the simulation at several sizes -- say 16, 32, 64, and 128 needles on a side -- and use a mathematical trick called finite-size scaling to extrapolate what an infinitely large magnet would do.

Imagine measuring how fast a rumor spreads in classrooms of 10, 20, 50, and 100 students, then predicting how it would spread across an entire country. Same idea.

## Real Metals, Real Magnets

The Ising model is not just a toy. We fit it to two real materials:

- **Iron (BCC crystal):** Our best-fit coupling strength is J = 14.3 meV, giving a Curie temperature of 1043 K. That is the temperature where iron actually stops being ferromagnetic -- and it matches experiment.
- **Nickel (FCC crystal):** J = 5.6 meV, Curie temperature around 631 K. Again, right in the ballpark of measured values.

So the same simple "match your neighbors" rule, tuned with one number per material, reproduces real-world magnetism.

## Freeze! (But Slowly)

Here is a fun one. If you cool a magnet through Tc very quickly, different regions of the magnet independently decide to align -- but they do not all pick the same direction. You end up with patches pointing up and patches pointing down, separated by walls of confused needles called *defects*.

This is the Kibble-Zurek mechanism, originally proposed to explain defects formed in the early universe after the Big Bang. The number of defects depends on how fast you cool: cool slower, fewer defects. We measured the scaling law and got an exponent of about 0.25, close to the theoretical prediction of 0.279. The universe and your refrigerator magnet obey the same rule.

## Poking Holes

What happens if you randomly snip some of the connections between neighbors? This is called bond dilution, and it models real-world impurities -- no crystal is perfect. We find that removing connections weakens the magnet and lowers Tc in a smooth, predictable way that matches theoretical expectations. The magnet degrades gracefully, not catastrophically, until you remove so many bonds that the grid falls apart.

## Teaching a Computer to See Phases

We also pointed a simple machine-learning model at raw snapshots of the needle grid and asked: "Can you figure out, without being told any physics, whether this snapshot is from a magnet or a non-magnet?" It can. The neural network learns to detect the phase transition on its own, and its answer agrees with our physics-based analysis. It is like showing someone photos of water and ice and having them deduce that 0 degrees Celsius is special -- without ever telling them what temperature means.

## Why Rust + GPU?

Monte Carlo simulations flip needles one at a time (or in clever batches) and repeat billions of times. Two things matter: speed per flip and not crashing after running for hours.

- **Rust** gives us C-level speed with memory safety guarantees -- no segfaults at 3 AM.
- **CUDA on an NVIDIA GPU** lets us flip thousands of needles simultaneously using a checkerboard pattern (color the grid like a chessboard; all the "black" squares can be flipped at the same time because none of them are neighbors).

The result: large simulations that would take days on a single CPU core finish in hours.

## How Good Are We?

| Quantity | Our Value | Reference | Error |
|---|---|---|---|
| 3D Tc | 4.5115 J/kB | 4.5115 (Hasenbusch 2010) | ~0.01% |
| 2D Tc | 2.269 J/kB | 2.2692 (Onsager exact) | <0.02% |
| KZ exponent | ~0.25 | 0.279 (theory) | ~10% |
| Fe Curie temp | 1043 K | 1043 K (experiment) | <1% |
| Ni Curie temp | ~631 K | ~627 K (experiment) | ~1% |

## What's Next

- Larger GPU runs on bigger lattices to sharpen the Kibble-Zurek exponent.
- More materials beyond iron and nickel.
- Explore disordered and frustrated magnets (spin glasses), where the needles receive conflicting instructions from their neighbors.
- Publication targeting Physical Review E, so other physicists can build on this work.

If you have read this far, you now understand -- at least in broad strokes -- what a research-grade Monte Carlo magnet simulator does and why anyone would bother building one. The short answer: because a billion tiny compass needles, following one simple rule, can teach us how order emerges from chaos.
